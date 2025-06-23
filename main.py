

import os
import shutil
import subprocess
from pathlib import Path
from tracker import MultiModelTracker

def clear_folder_contents(folder_path: Path):
    if not folder_path.exists():
        return
    for item in folder_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def main():
   
    project_root = Path(__file__).parent.resolve()
    
    folders_to_clear = [
        project_root / 'output_video' / 'ball_detection',
        project_root / 'tennis_statistics' / 'ball_stats',
        project_root / 'output_video' / 'tracked_everything',
        project_root / 'tennis_statistics' / 'stats_files',
        project_root / 'output_video' / 'segmentation_keypoint_detection',
    ]
 
    for folder in folders_to_clear:
        clear_folder_contents(folder)
        print(f"Limpiado: {folder}")

    
    seg_model_path = project_root / 'models_weigths' / 'segmentation' / 'best.pt'
    kp_model_path  = project_root / 'models_weigths' / 'keypoint_detection' / 'best.pt'
    input_video_path  = project_root / 'input_video' / 'videoprueba.mp4' 
    output_video_path = project_root / 'output_video' / 'segmentation_keypoint_detection' / 'tracked_video.avi'

    tracker = MultiModelTracker(str(seg_model_path), str(kp_model_path), extract_stats=True)
    tracker.track_video(str(input_video_path), str(output_video_path), fps=30)
    print("Tracking completado. Vídeo guardado en:", output_video_path)


    
    infer_script = project_root / 'tracknetV2' / 'infer_on_video.py'
    infer_cmd = [
        shutil.which('python') or 'python',
        str(infer_script),
        '--model_path', str(project_root / 'models_weigths' / 'object_detection' / 'model_best.pt'),
        '--video_path', str(project_root / 'output_video' / 'segmentation_keypoint_detection' / 'tracked_video.avi'),
        '--video_out_path', str(project_root / 'output_video' / 'tracked_everything' / 'final_tracked_video.avi'),
    ]
    print("Ejecutando:", ' '.join(infer_cmd))
    subprocess.run(infer_cmd, check=True)
    print("Inferencia completada.")

if __name__ == '__main__':
    main()
