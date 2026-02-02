import numpy as np
import matplotlib.pyplot as plt

def a2rad(angle):
    result = angle*np.pi / 180
    return result

def SinApp(angle):
    result = np.sin(a2rad(angle))
    return result

def RadApp(angle):
    result = angle * np.pi / 180
    return result

angle = np.linspace(-90, 90, 500)

Sin = []
rad = []
err = []

for i in angle:
    Sin.append(SinApp(i))
    rad.append(RadApp(i))
    err.append((RadApp(i) - SinApp(i))/SinApp(i))

plt.figure(figsize=(8, 4))
plt.plot(angle, Sin, label='Sin(x)', color='blue')
plt.plot(angle, rad, label='Rad', color='red')
plt.plot(angle, err, label='Error [%]', color='black')

plt.title("Sin(x) vs Radians")
plt.legend()
plt.show()