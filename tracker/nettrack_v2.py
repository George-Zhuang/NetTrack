import cv2
import torch
import numpy as np
import os

from tracker.kalman_filter import KalmanFilter
from tracker import matching
from tracker.basetrack import BaseTrack, TrackState
from tracker.visualize import save_result

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.traj = None
    
    def grid_sampling(self, grid=(3, 3), stride=8):
        x1, y1, x2, y2 = self.tlbr
        t = np.array((self.frame_id - 1) % (stride - 1))
        # sampling within the box 
        xs = np.linspace(x1, x2, int(grid[0]) + 2)
        ys = np.linspace(y1, y2, int(grid[1]) + 2)
        xs = xs[1:-1]
        ys = ys[1:-1]
        xs, ys = np.meshgrid(xs, ys)
        ts = t.repeat(grid[0]*grid[1])
        self.sampled_point = np.vstack((ts, xs.flatten(), ys.flatten())).T

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, stride, 
                 frame_id, grid=(3, 3), trajectories=None, 
                 visibilities=None, mode='coarse'):
        """ Start a new tracklet """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        # if mode == 'coarse':
        #     self.grid_sampling(grid, stride)
        self.grid_sampling(grid, stride)
        # elif mode == 'fine':
        if mode == 'fine':
            self.point_within_tlwh(trajectories, visibilities, stride, grid=grid)
    
    def point_within_tlwh(self, trajectories, visibilities, stride, grid=(3, 3)):
        ''' assign trajectory and visibility with the tracklet after activated '''
        if trajectories == None or visibilities == None:
            return None
        t_idx = (self.frame_id - 1) % (stride - 1) # In stack [1, 2, ..., 8], if activated at frame 2, t_idx = 1; In stack [8, 9, ..., 15], if activated at frame 9, t_idx = 1; ...
        within_mask = (trajectories[0, t_idx, :, 0] > self.tlwh[0]) & \
                      (trajectories[0, t_idx, :, 0] < self.tlwh[0]+self.tlwh[2]) & \
                      (trajectories[0, t_idx, :, 1] > self.tlwh[1]) & \
                      (trajectories[0, t_idx, :, 1] < self.tlwh[1]+self.tlwh[3])
        self.traj = trajectories[0, :, within_mask, :].detach().cpu().numpy()
        self.vis = visibilities[0, :, within_mask].detach().cpu().numpy()
        default_num_points = grid[0] * grid[1]
        if self.traj.shape[1] < default_num_points:
            self.traj = np.pad(self.traj, ((0, 0), (0, default_num_points - self.traj.shape[1]), (0, 0)), mode='constant', constant_values=-1)
            self.vis = np.pad(self.vis, ((0, 0), (0, default_num_points - self.vis.shape[1])), mode='constant', constant_values=0)
        elif self.traj.shape[1] > default_num_points:
            ind = np.random.choice(self.traj.shape[1], default_num_points, replace=False)
            self.traj = self.traj[:, ind, :]
            self.vis = self.vis[:, ind]

    def re_activate(self, new_track, frame_id, stride, grid=(3, 3), trajectories=None, 
                 visibilities=None, mode='coarse', new_id=False,):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self._tlwh = new_track.tlwh
        self.grid_sampling(grid, stride)
        if mode == 'fine':
            self.point_within_tlwh(trajectories, visibilities, stride, grid=grid)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self._tlwh = new_track.tlwh
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def next_points(self):
        stride = self.traj.shape[0] 
        next_idx = self.frame_id % stride
        return self.traj[next_idx]
    @property
    def curr_points(self):
        if self.traj is not None:
            stride = self.traj.shape[0] 
            curr_idx = (self.frame_id - 1) % stride
            return self.traj[curr_idx]
        else:
            return self.sampled_point[:, 1:]
    @property
    def next_vis(self):
        stride = self.vis.shape[0]
        next_idx = self.frame_id % stride
        return self.vis[next_idx]
    @property
    def curr_vis(self):
        stride = self.vis.shape[0]
        curr_idx = (self.frame_id - 1) % stride
        return self.vis[curr_idx]
    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        # if self.mean is None:
        #     return self._tlwh.copy()
        # ret = self.mean[:4].copy()
        # ret[2] *= ret[3]
        # ret[:2] -= ret[2:] / 2
        # return ret
        return self._tlwh.copy()

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class BaseTracker():
    def __init__(
            self, args, track_mode='coarse', 
            stride=8, grid=[3,3], frame_rate=30, track_buffer=30,
            score_mode='quadratic', match_thresh=0.95, 
            match_thresh_2=0.7, match_thresh_3=0.7, weight_pt=0.5):
        # base tracker info
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.args = args
        self.track_mode = track_mode
        self.stride = stride
        self.grid = grid
        self.track_buffer = track_buffer
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        # track and detection threshold
        if self.track_mode == 'coarse':
            self.det_thresh = args.track_thresh
        elif self.track_mode == 'fine':
            self.det_thresh = max(args.track_thresh - 0.1, 0.01) if not hasattr(args, 'det_thresh') else args.det_thresh
        # matching threshold
        self.score_mode = score_mode if not hasattr(args, 'score_mode') else args.score_mode
        self.match_thresh = match_thresh if not hasattr(args, 'match_thresh') else args.match_thresh
        self.match_thresh_2 = match_thresh_2 if not hasattr(args, 'match_thresh_2') else args.match_thresh_2
        self.match_thresh_3 = match_thresh_3 if not hasattr(args, 'match_thresh_3') else args.match_thresh_3
        self.weight_pt = weight_pt if not hasattr(args, 'weight_pt') else args.weight_pt

    def empty_tracks(self):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()

    def predict(self, output_results, trajectories=None, visibilities=None):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results) == 0:
            scores = np.array([])
            bboxes = np.array([])
        elif output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.01
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        if self.track_mode == 'coarse':
            dists = matching.iou_distance(strack_pool, detections)
        elif self.track_mode == 'fine':
            dists = matching.point_within_box_distance(strack_pool, detections, mode=self.score_mode, weight_pt=self.weight_pt)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh,)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, stride=self.stride,
                                  grid=self.grid, trajectories=trajectories,
                                  visibilities=visibilities, mode=self.track_mode, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if self.track_mode == 'coarse':
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
        elif self.track_mode == 'fine':
            dists = matching.point_within_box_distance(r_tracked_stracks, detections_second, mode=self.score_mode, weight_pt=self.weight_pt)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.match_thresh_2)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, stride=self.stride,
                                  grid=self.grid, trajectories=trajectories,
                                  visibilities=visibilities, mode=self.track_mode, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        if self.track_mode == 'coarse':
            dists = matching.iou_distance(unconfirmed, detections)
        elif self.track_mode == 'fine':
            dists = matching.point_within_box_distance(unconfirmed, detections, mode=self.score_mode, weight_pt=self.weight_pt)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh_3)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.stride, self.frame_id,
                           grid=self.grid, trajectories=trajectories,
                           visibilities=visibilities, mode=self.track_mode)
            
            activated_stracks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        if self.track_mode == 'fine':
            output_stracks = [track for track in self.tracked_stracks if track.is_activated]
            return output_stracks

class NetTracker():
    def __init__(self, args, point_tracker, stride=8, grid=[3,3]):
        self.coarse_tracker = BaseTracker(args, track_mode='coarse', stride=stride, grid=grid)
        self.fine_tracker = BaseTracker(args, track_mode='fine', stride=stride, grid=grid)
        self.point_tracker = point_tracker
        self.args = args
        self.stride = stride if not hasattr(args, 'stride') else args.stride
        self.grid = grid if not hasattr(args, 'grid') else args.grid

    def inference(self, det_preds, img_paths, seq_name):
        ''' Step 0: Initialization '''
        self.coarse_tracker.predict(np.array(det_preds[0]))
        self.fine_tracker.predict(np.array(det_preds[0]))
        # self.fine_tracker.tracked_stracks = self.coarse_tracker.tracked_stracks
        frame = cv2.imread(img_paths[0])
        rgb_stack = [np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))]
        det_stack = [det_preds[0]]
        img_path_stack = [img_paths[0]]
        results = save_result(
            img_path=img_paths[0], 
            stracks=self.fine_tracker.tracked_stracks,
            frame_id=1, 
            seq_name=seq_name,
            output_path=self.args.output_path,
            vis_box=self.args.vis_box,
            vis_points=self.args.vis_points,
            min_area=self.args.min_area,
            max_area=self.args.max_area
            )
        
        for frame_id, (img_path, det_pred) in enumerate(zip(img_paths[1:], det_preds[1:]), 2):
            ''' Step I: Coarse prediction '''
            self.coarse_tracker.predict(np.array(det_pred))
            det_stack.append(det_pred)
            img_path_stack.append(img_path)
            frame = cv2.imread(img_path)
            rgb_stack.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            if frame_id % (self.stride-1) == 1 or frame_id == len(img_paths):
                ''' Step II: Multi-granular learning '''
                # get frame stack
                while len(rgb_stack) < self.stride // 2 + 1:
                    rgb_stack.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                rgb_stack = np.stack(rgb_stack)
                rgb_stack = torch.from_numpy(rgb_stack).permute(0, 3, 1, 2)[None].float().to(self.args.device)
                # point query
                query = []
                for track in self.coarse_tracker.tracked_stracks:
                    query.append(track.sampled_point)    
                if len(query) > 0:
                    query = np.stack(query).reshape(1, -1, 3)
                    query = torch.from_numpy(query).float().to(self.args.device)
                else:
                    query = None
                with torch.no_grad():
                    trajectories, visibilities = self.point_tracker(rgb_stack, query, grid_size=10)
                for track in self.fine_tracker.tracked_stracks:
                    track.point_within_tlwh(trajectories, visibilities, stride=self.stride)
                for idx, (det_pred, img_path) in enumerate(zip(det_stack[1:], img_path_stack[1:]), 1):
                    output_stracks = self.fine_tracker.predict(np.array(det_pred),
                                                               trajectories=trajectories,
                                                               visibilities=visibilities
                                                               )

                    stack_frame_id = frame_id - len(img_path_stack) + idx + 1
                    extra_points = trajectories[:, idx, :, :].detach().cpu().numpy()
                    result = save_result(img_path=img_path, 
                                         stracks=output_stracks,
                                         frame_id=stack_frame_id, 
                                         seq_name=seq_name,
                                         output_path=self.args.output_path,
                                         vis_box=self.args.vis_box,
                                         vis_points=self.args.vis_points,
                                         extra_points=extra_points,
                                         min_area=self.args.min_area,
                                         max_area=self.args.max_area
                                         )
                    if result is not None:
                        results += result
                for track in self.fine_tracker.tracked_stracks:
                    track.grid_sampling(grid=self.grid)
                self.coarse_tracker.tracked_stracks = self.fine_tracker.tracked_stracks
                self.coarse_tracker.lost_stracks = self.fine_tracker.lost_stracks
                self.coarse_tracker.removed_stracks = self.fine_tracker.removed_stracks
                rgb_stack = [np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))]
                det_stack = [det_pred]
                img_path_stack = [img_path]
        self.fine_tracker.empty_tracks()
        self.coarse_tracker.empty_tracks()
        return results

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
