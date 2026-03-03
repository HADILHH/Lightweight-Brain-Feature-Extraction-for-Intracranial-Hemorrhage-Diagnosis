import os
import matplotlib.pyplot as plt

train_dir = "data/train"
test_dir = "data/test"

def count_images(folder):
    count = 0
    for root, dirs, files in os.walk(folder):
        count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return count

train_count = count_images(train_dir)
test_count = count_images(test_dir)
total = train_count + test_count

labels = [
    f'Training Set\n({train_count} images)',
    f'Testing Set\n({test_count} images)'
]
sizes = [train_count, test_count]

colors = ['#9b59b6', '#d2b4de']

plt.figure(figsize=(6,6))
plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct=lambda p: f'{p:.1f}%\n({int(p*total/100)})',
    startangle=90,
    explode=(0.05, 0)
)

plt.title('Distribution of the Dataset')
plt.tight_layout()
plt.show()
