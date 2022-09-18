import matplotlib.pyplot as plt
import pandas as pd
from Model_Logistic import logistic_model
from Model_DT import DT_model
from Model_RF import RF_model
from Model_GaussianNB import GaussianNB_model
from Model_GB import GB_model
import settingFunction as sf

block = pd.read_csv(
    "C:\\Users\\wxy\\PycharmProjects\\project108\\AVFflow\\0_test.csv")
normal = pd.read_csv(
    "C:\\Users\\wxy\\PycharmProjects\\project108\\AVFflow\\1_test.csv")

fig, axes = plt.subplots(1, 2)
ax_val = axes[0]
ax_test = axes[1]

logistic_model(block, normal, ax_val, ax_test)
DT_model(block, normal, ax_val, ax_test)
RF_model(block, normal, ax_val, ax_test)
GaussianNB_model(block, normal, ax_val, ax_test)
GB_model(block, normal, ax_val, ax_test)

plt.show()