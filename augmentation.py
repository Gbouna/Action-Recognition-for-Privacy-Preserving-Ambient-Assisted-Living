import numpy as np
import random
import math

class Augmentation:
    def jittering_all_frames_all_joints(self, skeleton_data, sigma=0.5):
        """
        Apply jittering augmentation to skeleton data.
        Parameters:
        - skeleton_data: np.array of shape (num_frames, num_joints, 3)
        - sigma: Standard deviation of Gaussian noise
        Returns:
        - Augmented skeleton data
        """
        noise = np.random.normal(0, sigma, skeleton_data.shape)
        augmented_data = skeleton_data + noise
        return augmented_data

    def jittering_all_frames_specific_joints(self, skeleton_data, sigma=0.5):
        """
        Apply jittering augmentation to specific joints in skeleton data.
        Parameters:
        - skeleton_data: np.array of shape (num_frames, num_joints, 3)
        - sigma: Standard deviation of Gaussian noise
        Returns:
        - Augmented skeleton data
        """
        augmented_data = np.copy(skeleton_data)
        num_frames = skeleton_data.shape[0]
        joints_to_add_noise = [3, 4, 5, 11, 12, 13, 7, 8, 9, 15, 16, 17]
        for i in range(num_frames):
            for joint in joints_to_add_noise:
                noise = np.random.normal(0, sigma, 3)  
                augmented_data[i, joint] += noise  
        return augmented_data

    def jittering_all_frames_varying_magnitude_between_joint(self, skeleton_data, sigma_high=0.5, sigma_low=0.05):
        """
        Apply jittering augmentation to skeleton data with different noise magnitudes for specific joints.
        Parameters:
        - skeleton_data: np.array of shape (num_frames, num_joints, 3)
        - sigma_high: Higher standard deviation of Gaussian noise for specific joints
        - sigma_low: Lower standard deviation of Gaussian noise for other joints
        Returns:
        - Augmented skeleton data
        """
        augmented_data = np.copy(skeleton_data)
        num_frames, num_joints = skeleton_data.shape[:2]
        joints_to_add_high_noise = [3, 4, 5, 11, 12, 13, 7, 8, 9, 15, 16, 17]
        for i in range(num_frames):
            for j in range(num_joints):
                sigma = sigma_high if j in joints_to_add_high_noise else sigma_low
                noise = np.random.normal(0, sigma, 3) 
                augmented_data[i, j] += noise  
        return augmented_data

    def jittering_random_frames_all_joints(self, skeleton_data, sigma=0.5, frame_probability=0.5):
        """
        Apply jittering augmentation to skeleton data with noise added to random frames to all  joints.
        Parameters:
        - skeleton_data: np.array of shape (num_frames, num_joints, 3)
        - sigma: Standard deviation of Gaussian noise
        - frame_probability: Probability of a frame to be selected for noise addition
        Returns:
        - Augmented skeleton data
        """
        augmented_data = np.copy(skeleton_data)
        num_frames = skeleton_data.shape[0]
        for i in range(num_frames):
            if np.random.rand() < frame_probability:
                noise = np.random.normal(0, sigma, skeleton_data[i].shape)
                augmented_data[i] += noise
        return augmented_data

    def jittering_random_frames_specific_joints(self, skeleton_data, sigma=0.5, frame_probability=0.5):
        """
        Apply jittering augmentation to skeleton data with noise added to specific joints in random frames.
        Parameters:
        - skeleton_data: np.array of shape (num_frames, num_joints, 3)
        - sigma: Standard deviation of Gaussian noise
        - frame_probability: Probability of a frame to be selected for noise addition
        Returns:
        - Augmented skeleton data
        """
        augmented_data = np.copy(skeleton_data)
        num_frames = skeleton_data.shape[0]
        joints_to_add_noise = [3, 4, 5, 11, 12, 13, 7, 8, 9, 15, 16, 17]
        frames_with_noise = []
        for i in range(num_frames):
            if np.random.rand() < frame_probability:
                for joint in joints_to_add_noise:
                    noise = np.random.normal(0, sigma, 3)  
                    augmented_data[i, joint] += noise  
                frames_with_noise.append(i)
        return augmented_data

    def jittering_random_frames_varying_magnitude_between_joint(self, skeleton_data, sigma_high=0.5, sigma_low=0.05, frame_probability=0.5):
        """
        Apply jittering augmentation to skeleton data with noise added to specific joints in random frames,
        using different noise magnitudes for specific joints.
        Parameters:
        - skeleton_data: np.array of shape (num_frames, num_joints, 3)
        - sigma_high: Higher standard deviation of Gaussian noise for specific joints
        - sigma_low: Lower standard deviation of Gaussian noise for other joints
        - frame_probability: Probability of a frame to be selected for noise addition
        Returns:
        - Augmented skeleton data
        """
        augmented_data = np.copy(skeleton_data)
        num_frames, num_joints = skeleton_data.shape[:2]
        joints_to_add_high_noise = [3, 4, 5, 11, 12, 13, 7, 8, 9, 15, 16, 17]
        frames_with_noise = []
        for i in range(num_frames):
            if np.random.rand() < frame_probability:
                for j in range(num_joints):
                    sigma = sigma_high if j in joints_to_add_high_noise else sigma_low
                    noise = np.random.normal(0, sigma, 3) 
                    augmented_data[i, j] += noise  
                frames_with_noise.append(i)
        return augmented_data

    def jittering_random_consecutive_frames_all_joints(self, skeleton_data, sigma=0.5, min_sequence_length=5, max_sequence_length=10):
        """
        Apply jittering augmentation to skeleton data with noise added to random continuous frames,
        with random skips between the noisy sequences.
        Parameters:
        - skeleton_data: np.array of shape (num_frames, num_joints, 3)
        - sigma: Standard deviation of Gaussian noise
        - min_sequence_length: Minimum length of continuous frame sequence with noise
        - max_sequence_length: Maximum length of continuous frame sequence with noise
        Returns:
        - Augmented skeleton data
        """
        augmented_data = np.copy(skeleton_data)
        num_frames = skeleton_data.shape[0]
        frames_with_noise = []  
        current_frame = 0
        while current_frame < num_frames:
            remaining_frames = num_frames - current_frame
            max_possible_seq_length = min(max_sequence_length, remaining_frames)
            if remaining_frames < min_sequence_length:
                break
            sequence_length = np.random.randint(min_sequence_length, max_possible_seq_length + 1)
            for i in range(current_frame, current_frame + sequence_length):
                noise = np.random.normal(0, sigma, skeleton_data[i].shape)
                augmented_data[i] += noise
                frames_with_noise.append(i)
            current_frame += sequence_length
            if current_frame < num_frames:
                remaining_frames = num_frames - current_frame
                if remaining_frames < 3:
                    skip_length = remaining_frames 
                elif remaining_frames == 3:
                    skip_length = 3  
                else:
                    skip_length = np.random.randint(3, remaining_frames) 
                current_frame += skip_length
        return augmented_data

    def jittering_random_consecutive_frames_sepecific_joints(self, skeleton_data, sigma=0.5, min_sequence_length=5, max_sequence_length=10):
        """
        Apply jittering augmentation to specific joints in skeleton data with noise added to random continuous frames,
        with random skips between the noisy sequences.
        Parameters:
        - skeleton_data: np.array of shape (num_frames, num_joints, 3)
        - sigma: Standard deviation of Gaussian noise
        - min_sequence_length: Minimum length of continuous frame sequence with noise
        - max_sequence_length: Maximum length of continuous frame sequence with noise
        Returns:
        - Augmented skeleton data
        """
        augmented_data = np.copy(skeleton_data)
        num_frames = skeleton_data.shape[0]
        joints_to_add_noise = [3, 4, 5, 11, 12, 13, 7, 8, 9, 15, 16, 17]  
        frames_with_noise = []  
        current_frame = 0
        while current_frame < num_frames:
            remaining_frames = num_frames - current_frame
            max_possible_seq_length = min(max_sequence_length, remaining_frames)
            if remaining_frames < min_sequence_length:
                break
            sequence_length = np.random.randint(min_sequence_length, max_possible_seq_length + 1)
            for i in range(current_frame, current_frame + sequence_length):
                for joint in joints_to_add_noise:
                    noise = np.random.normal(0, sigma, 3)  
                    augmented_data[i, joint] += noise  
                frames_with_noise.append(i)
            current_frame += sequence_length
            if current_frame < num_frames:
                remaining_frames = num_frames - current_frame
                if remaining_frames < 3:
                    skip_length = remaining_frames  
                elif remaining_frames == 3:
                    skip_length = 3  
                else:
                    skip_length = np.random.randint(3, remaining_frames)  
                current_frame += skip_length
        return augmented_data

    def jittering_random_consecutive_frames_varying_magnitude_between_joints(self, skeleton_data, sigma_high=0.5, sigma_low=0.05, min_sequence_length=5, max_sequence_length=10):
        """
        Apply jittering augmentation to skeleton data with different noise magnitudes for specific joints in random continuous frames,
        with random skips between the noisy sequences.
        Parameters:
        - skeleton_data: np.array of shape (num_frames, num_joints, 3)
        - sigma_high: Higher standard deviation of Gaussian noise for specific joints
        - sigma_low: Lower standard deviation of Gaussian noise for other joints
        - min_sequence_length: Minimum length of continuous frame sequence with noise
        - max_sequence_length: Maximum length of continuous frame sequence with noise
        Returns:
        - Augmented skeleton data
        """
        augmented_data = np.copy(skeleton_data)
        num_frames, num_joints = skeleton_data.shape[:2]
        joints_to_add_high_noise = [3, 4, 5, 11, 12, 13, 7, 8, 9, 15, 16, 17]
        frames_with_noise = []  
        current_frame = 0
        while current_frame < num_frames:
            remaining_frames = num_frames - current_frame
            max_possible_seq_length = min(max_sequence_length, remaining_frames)
            if remaining_frames < min_sequence_length:
                break
            sequence_length = np.random.randint(min_sequence_length, max_possible_seq_length + 1)
            for i in range(current_frame, current_frame + sequence_length):
                for j in range(num_joints):
                    sigma = sigma_high if j in joints_to_add_high_noise else sigma_low
                    noise = np.random.normal(0, sigma, 3)  
                    augmented_data[i, j] += noise  
                frames_with_noise.append(i)
            current_frame += sequence_length
            if current_frame < num_frames:
                remaining_frames = num_frames - current_frame
                if remaining_frames < 3:
                    skip_length = remaining_frames
                elif remaining_frames == 3:
                    skip_length = 3
                else:
                    skip_length = np.random.randint(3, remaining_frames)
                current_frame += skip_length
        return augmented_data

    def occluding_specific_joints_all_frames(self, data, s_l, s_h, v_l, v_h, joints_to_erase=[2, 3, 4, 5]):
        """
        Apply occlusion augmentation to specific joints in skeleton data.
        Parameters:
        - skeleton_data: np.array of shape (num_frames, num_joints, 3)
        - sigma: Standard deviation of Gaussian noise
        Returns:
        - Augmented skeleton data
        """
        num_frames, num_keypoints, _ = data.shape
        area = num_frames * len(joints_to_erase)
        erased_joints_info = []  
        se = np.random.uniform(s_l, s_h) * area
        fe = num_frames  
        for joint in joints_to_erase:
            if joint < num_keypoints:
                data[:, joint, :] = np.random.uniform(v_l, v_h, (fe, 3))
                erased_joints_info.append((joint, 0, fe)) 
        return data

    def occluding_specific_joints_random_consecutive_frames(self, data, p, s_l, s_h, v_l, v_h, min_sequence_length=5, max_sequence_length=10, joints_to_erase=[7, 8, 9, 15, 16, 17]):
        if np.random.rand() > p:
            return data
        num_frames, num_keypoints, _ = data.shape
        area = num_frames * len(joints_to_erase)
        erased_frames_indices = []
        current_frame = 0
        while current_frame < num_frames:
            remaining_frames = num_frames - current_frame
            max_possible_seq_length = min(max_sequence_length, remaining_frames)
            if remaining_frames < min_sequence_length:
                break
            sequence_length = np.random.randint(min_sequence_length, max_possible_seq_length + 1)
            for _ in range(100):
                se = np.random.uniform(s_l, s_h) * area
                re = np.random.uniform(1, 1)  
                fe = int(np.sqrt(se * re))
                if fe >= num_frames:
                    continue
                max_start_frame = min(current_frame + sequence_length - fe, num_frames - fe)
                if current_frame >= max_start_frame:
                    continue
                ye = np.random.randint(current_frame, max_start_frame)
                if ye + fe <= num_frames:
                    for joint in joints_to_erase:
                        if joint < num_keypoints:
                            data[ye:ye+fe, joint, :] = np.random.uniform(v_l, v_h, (fe, 3))
                    erased_frames_indices.extend(range(ye, ye+fe))
                    break
            current_frame += sequence_length
            if current_frame < num_frames:
                remaining_frames = num_frames - current_frame
                skip_length = remaining_frames if remaining_frames <= 3 else np.random.randint(3, remaining_frames)
                current_frame += skip_length
        return data

    def occluding_random_joints_random_consecutive_frames(self, data, p, s_l, s_h, r_1, r_2, v_l, v_h, min_sequence_length=5, max_sequence_length=10):
        if np.random.rand() > p:
            return data
        num_frames, num_keypoints, _ = data.shape
        area = num_frames * num_keypoints
        erased_frames_indices = []  
        current_frame = 0
        while current_frame < num_frames:
            remaining_frames = num_frames - current_frame
            max_possible_seq_length = min(max_sequence_length, remaining_frames)
            if remaining_frames < min_sequence_length:
                break
            sequence_length = np.random.randint(min_sequence_length, max_possible_seq_length + 1)
            for _ in range(100):
                se = np.random.uniform(s_l, s_h) * area
                re = np.random.uniform(r_1, r_2)
                fe = int(np.sqrt(se * re))
                ke = int(np.sqrt(se / re))
                if fe >= num_frames or ke >= num_keypoints:
                    continue
                xe = np.random.randint(0, num_keypoints - ke)
                max_start_frame = min(current_frame + sequence_length - fe, num_frames - fe)
                if current_frame >= max_start_frame:
                    continue
                ye = np.random.randint(current_frame, max_start_frame)
                if xe + ke <= num_keypoints and ye + fe <= num_frames:
                    data[ye:ye+fe, xe:xe+ke, :] = np.random.uniform(v_l, v_h, (fe, ke, 3))
                    erased_frames_indices.extend(range(ye, ye+fe))
                    break
            current_frame += sequence_length
            if current_frame < num_frames:
                remaining_frames = num_frames - current_frame
                if remaining_frames <= 3:
                    skip_length = remaining_frames  
                else:
                    skip_length = np.random.randint(3, remaining_frames)  
                current_frame += skip_length
        return data

    def occluding_all_joints_random_consecutive_frames(self, data, p, s_l, s_h, v_l, v_h, min_sequence_length=5, max_sequence_length=10):
        if np.random.rand() > p:
            return data
        num_frames, num_keypoints, _ = data.shape
        area = num_frames * num_keypoints
        erased_frames_indices = []
        current_frame = 0
        while current_frame < num_frames:
            remaining_frames = num_frames - current_frame
            max_possible_seq_length = min(max_sequence_length, remaining_frames)
            if remaining_frames < min_sequence_length:
                break
            sequence_length = np.random.randint(min_sequence_length, max_possible_seq_length + 1)
            for _ in range(100):
                se = np.random.uniform(s_l, s_h) * area
                re = np.random.uniform(1, 1)  
                fe = int(np.sqrt(se * re))
                if fe >= num_frames:
                    continue
                max_start_frame = min(current_frame + sequence_length - fe, num_frames - fe)
                if current_frame >= max_start_frame:
                    continue
                ye = np.random.randint(current_frame, max_start_frame)
                if ye + fe <= num_frames:
                    data[ye:ye+fe, :, :] = np.random.uniform(v_l, v_h, (fe, num_keypoints, 3))
                    erased_frames_indices.extend(range(ye, ye+fe))
                    break
            current_frame += sequence_length
            if current_frame < num_frames:
                remaining_frames = num_frames - current_frame
                skip_length = remaining_frames if remaining_frames <= 3 else np.random.randint(3, remaining_frames)
                current_frame += skip_length
        return data