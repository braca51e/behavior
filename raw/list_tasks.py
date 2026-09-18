import json
tasks = []
for line in open('/home/luis/behavior_challenge/raw/tasks.jsonl'):
    o = json.loads(line)
    tasks.append((o['task_index'], o['task_name'], o['task']))
print('total tasks:', len(tasks))
with open('/home/luis/behavior_challenge/raw/tasks_100.txt', 'w') as f:
    for i, name, desc in tasks:
        f.write(f"{i:3d}  {name}\n")
# print a sample
for i, name, desc in tasks[:12]:
    print(f"{i:3d}  {name}")
print('...')
for i, name, desc in tasks[-3:]:
    print(f"{i:3d}  {name}")
