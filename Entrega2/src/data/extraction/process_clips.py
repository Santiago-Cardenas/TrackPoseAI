"""
Script para procesar clips de Entrega 2
"""

from pathlib import Path
import pandas as pd
import sys

# Importar desde la subcarpeta
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'data' / 'extraction'))
from body_tracker import BodyTracker

class MotionAnalyzerV2:
    def __init__(self):
        self.body_tracker = BodyTracker()

    def analyze_all_clips(self, source_folder, destination_csv):
        """Analiza todos los clips MP4 recursivamente."""
        folder_path = Path(source_folder)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        collected_data = []
        clip_files = list(folder_path.rglob('*.mp4'))

        if not clip_files:
            raise ValueError(f"No MP4 files found in: {folder_path}")

        print(f"\n{'='*70}")
        print(f"PROCESANDO CLIPS DE ENTREGA 2")
        print(f"{'='*70}")
        print(f"Archivos encontrados: {len(clip_files)}")
        print(f"{'='*70}\n")

        for i, clip_file in enumerate(clip_files, 1):
            print(f"[{i}/{len(clip_files)}] {clip_file.name}")
            
            clip_dataframe = self.body_tracker.extract_landmarks_from_clip(clip_file)

            if clip_dataframe.empty:
                print(f"  ⚠️  No landmarks detectados")
                continue

            # Parsear nombre: Accion_Velocidad_Angulo_Distancia.mp4
            filename_segments = clip_file.stem.split('_')
            
            if len(filename_segments) >= 2:
                action = filename_segments[0].lower()
                speed = filename_segments[1].lower()
                angle = filename_segments[2] if len(filename_segments) > 2 else 'unknown'
                distance = filename_segments[3] if len(filename_segments) > 3 else 'unknown'
                
                # Mapear acciones
                action_mapping = {
                    'adelante': 'caminar_adelante',
                    'atras': 'caminar_atras', 
                    'atrás': 'caminar_atras',
                    'giro': 'girar',
                    'levantarse': 'levantarse',
                    'sentarse': 'sentarse',
                }
                
                movement_type = action_mapping.get(action, action)
                speed = 'rapido' if speed in ['rapido', 'rápido'] else speed
                
            else:
                movement_type = 'undefined'
                speed = 'undefined'
                angle = 'unknown'
                distance = 'unknown'

            # Agregar metadatos
            clip_dataframe['movement_type'] = movement_type
            clip_dataframe['speed'] = speed
            clip_dataframe['camera_angle'] = angle
            clip_dataframe['camera_distance'] = distance

            collected_data.append(clip_dataframe)
            print(f"  ✓ {len(clip_dataframe)} frames | {movement_type} | {speed}")

        if not collected_data:
            raise ValueError("No se pudo procesar ningún video")

        print(f"\n{'='*70}")
        print("CONSOLIDANDO...")
        
        consolidated_dataframe = pd.concat(collected_data, ignore_index=True)
        
        output_location = Path(destination_csv)
        output_location.parent.mkdir(parents=True, exist_ok=True)
        consolidated_dataframe.to_csv(destination_csv, index=False)

        print(f"\n✓ Dataset guardado: {destination_csv}")
        print(f"\n{'='*70}")
        print("ESTADÍSTICAS")
        print(f"{'='*70}")
        print(f"Total frames: {len(consolidated_dataframe)}")
        print(f"Clips: {consolidated_dataframe['clip_name'].nunique()}")
        print(f"\nActividades:")
        print(consolidated_dataframe['movement_type'].value_counts())
        print(f"\nVelocidades:")
        print(consolidated_dataframe['speed'].value_counts())
        print(f"\nÁngulos:")
        print(consolidated_dataframe['camera_angle'].value_counts())
        print(f"\nDistancias:")
        print(consolidated_dataframe['camera_distance'].value_counts())
        print(f"{'='*70}\n")

        return consolidated_dataframe


def main():
    # Rutas relativas desde Entrega2/
    clips_folder = "../clips"
    output_csv = "../output/motion_data_entrega2.csv" 
    
    print("\nINICIANDO PROCESAMIENTO")
    print(f"Clips: {clips_folder}")
    print(f"Salida: {output_csv}\n")
    
    try:
        analyzer = MotionAnalyzerV2()
        df = analyzer.analyze_all_clips(clips_folder, output_csv)
        print(" COMPLETADO\n")
        return df
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
