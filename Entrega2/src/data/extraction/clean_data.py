import pandas as pd
   
df = pd.read_csv('../output/motion_data_entrega2.csv')
print(f"Antes: {len(df)} frames")
   
df['camera_angle'] = df['camera_angle'].str.strip()
df['camera_distance'] = df['camera_distance'].str.strip().str.replace('.mp4', '', regex=False)
   
df.to_csv('../output/motion_data_entrega2.csv', index=False)
print("CSV limpio guardado")