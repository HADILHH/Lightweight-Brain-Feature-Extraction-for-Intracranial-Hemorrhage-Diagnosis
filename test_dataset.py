import matplotlib.pyplot as plt

# عدد الملفات
train_count = 60
test_count = 15

# Labels و Values
labels = ['Training Set', 'Testing Set']
sizes = [train_count, test_count]
colors = ['skyblue', 'lightgreen']

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=(0.05, 0))
plt.title('Dataset Split')
plt.show()
