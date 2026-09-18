import json
info = json.load(open('/home/luis/behavior_challenge/raw/demo_info.json'))
print('=== FEATURES (observation/action schema) ===')
for k, v in info.get('features', {}).items():
    print(f"{k}: shape={v.get('shape')} dtype={v.get('dtype')}")
print()
print('total_episodes:', info.get('total_episodes'))
print('total_frames:', info.get('total_frames'))
print('total_tasks:', info.get('total_tasks'))
print('robot:', info.get('robot'))
print('chunks_size:', info.get('chunks_size'))
print('fps:', info.get('fps'))
print('codebase_version:', info.get('codebase_version'))
