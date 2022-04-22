import main_videos, main_subvideos, main_frames, main_faces, main_individuals, main_duplicates
from variables_and_constants import constants

def main():
    print(constants.CHALLENGE)
    main_videos.main_videos()
    main_subvideos.main_subvideos()
    main_frames.main_frames()
    main_faces.main_faces()
    main_individuals.main_individuals()
    main_duplicates.main_duplicates()

if __name__ == '__main__':
    main()
