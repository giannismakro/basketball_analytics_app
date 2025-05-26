import os
import argparse
from court_keypoint_detector import CourtKeypointDetector
from trackers.hoop_tracker import HoopTracker
from utils.video_utils import read_video, save_video, check_for_shots
import torch.serialization
from trackers import PlayerTracker, BallTracker
from torch.nn.modules.container import Sequential, ModuleList
from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF, Concat, Detect, DFL
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from team_assigner import TeamAssigner
from ball_aquisition import BallAcquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from ultralytics.nn.tasks import DetectionModel
from core.team import Team
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer,
    HoopTracksDrawer
)
from utils.report_generator import generate_game_summary_pdf
from configs import(
    STUBS_DEFAULT_PATH,
    HOOP_DETECTOR_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH
)

def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_PATH,
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                        help='Path to stub directory')
    return parser.parse_args()


torch.serialization.add_safe_globals([
    DetectionModel,
    Sequential,
    Upsample,
    Concat,
    Conv,
    Conv2d,
    BatchNorm2d,
    Detect,
    SiLU,
    C2f,
    DFL,
    ModuleList,
    MaxPool2d,
    Bottleneck,
    SPPF
])


def main():
    print("Basketball Video Analysis")
    args = parse_args()

    # Read Video
    print(f"Input video {args.input_video}")
    video_frames = read_video(args.input_video)
    print(f"Number of frames: {len(video_frames)}")
    ## Initialize Tracker

    team1 = Team("name1", "white shirt")
    team2 = Team("name2", "blue shirt")

    ## Initialize Keypoint Detector
    # We have different models hence why we predict lot of times
    # each model is speciliazed for different task
    print("Running Trackers")
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)
    # Run Detectors to get Player, Ball and Hoop Tracks lists for each frame
    players = PlayerTracker(PLAYER_DETECTOR_PATH).get_player_objects(video_frames)
    baskets = HoopTracker(HOOP_DETECTOR_PATH).get_tracks(video_frames)
    ball_object = BallTracker(BALL_DETECTOR_PATH).get_object_tracks(video_frames)
    court_keypoints_tracks = court_keypoint_detector.get_court_keypoints(video_frames)



    ## Draw object Tracks
    player_tracks_drawer = PlayerTracksDrawer()
    # Assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames,
                                                                    players,
                                                                    read_from_stub=False,
                                                                    stub_path=os.path.join(args.stub_path, 'player_assignment_stub.pkl')
                                                                    )

    # Ball Acquisition
    print("Ball acquisition")
    possession_list = BallAcquisitionDetector().detect_ball_possession(players, ball_object)
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(possession_list,player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(possession_list,player_assignment)

    # Detect Passes
    # Tactical View
    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png"
    )

    court_keypoints_tracks = tactical_view_converter.validate_keypoints(court_keypoints_tracks)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints_tracks, players)
    # Speed and Distance Calculator
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters
    )
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame)
    output_video_frames = player_tracks_drawer.draw(video_frames,
                                                    players,
                                                    player_assignment,
                                                    possession_list)

    # Draw output
    # Initialize Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    frame_number_drawer = FrameNumberDrawer()
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()

    ## Draw object Tracks
    output_video_frames = player_tracks_drawer.draw(video_frames,
                                                    players,
                                                    player_assignment,
                                                    possession_list)

    output_video_frames = check_for_shots(output_video_frames, ball_object, baskets, possession_list, players)

    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_object)

    ## Draw KeyPoints
    output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints_tracks)

    ## Draw Frame Number
    output_video_frames = frame_number_drawer.draw(output_video_frames)

    # Draw Team Ball Control
    output_video_frames = team_ball_control_drawer.draw(output_video_frames,
                                                        player_assignment,
                                                        possession_list)

    # Draw Passes and Interceptions
    output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames,
                                                             passes,
                                                             interceptions)

    # Speed and Distance Drawer
    output_video_frames = speed_and_distance_drawer.draw(output_video_frames,
                                                         players,
                                                         player_distances_per_frame,
                                                         player_speed_per_frame
                                                         )

    ## Draw Tactical View
    output_video_frames = tactical_view_drawer.draw(output_video_frames,
                                                    tactical_view_converter.court_image_path,
                                                    tactical_view_converter.width,
                                                    tactical_view_converter.height,
                                                    tactical_view_converter.key_points,
                                                    tactical_player_positions,
                                                    player_assignment,
                                                    possession_list,
                                                    )

    hoop_drawer = HoopTracksDrawer()
    output_video_frames = hoop_drawer.draw(output_video_frames, baskets[0], baskets[1])
    # Save video
    print(f"Saving video file {args.output_video}")
    save_video(output_video_frames, args.output_video)
    generate_game_summary_pdf("output/game_summary.pdf", teams=[team1, team2], players=players)


if __name__ == '__main__':
    main()
