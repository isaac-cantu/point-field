from pointfield.geometry.shapes2d.circle import Circle
import matplotlib.pyplot as plt

circle = Circle(radius=2.0)

print("Radius:")
print(circle.radius)

print("\nIntrinsic dimension:")
print(circle.intrinsic_dimension)

print("\nAmbient dimension:")
print(circle.ambient_dimension)

print("\nCentroid:")
print(circle.centroid)

print("\nBounds:")
print(circle.bounds)

X = circle.sample(
    n_points=1000,
    seed=42,
)

print("\nPoints:")
print(X)

print("\nShape:")
print(X.shape)

plt.scatter(X[:,0],X[:,1])
plt.show()