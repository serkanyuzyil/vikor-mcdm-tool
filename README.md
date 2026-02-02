# vikor-mcdm-tool
# VIKOR MCDM Tool for Material Selection

## Overview
This project is a Python-based decision support tool designed to solve the **"Flywheel Material Selection"** problem using Multi-Criteria Decision Making (MCDM) methods. It implements the **VIKOR** method to rank alternatives and utilizes **Entropy Weighting** to determine objective weights for the criteria.

This tool helps engineers and decision-makers select the optimal material by balancing conflicting criteria (e.g., cost, strength, density).

## Key Features
* **Entropy Method:** Automatically calculates objective weights for criteria based on data variance.
* **VIKOR Algorithm:** Implements the full VIKOR procedure to determine the compromise solution.
* **Data Integration:** Reads input data directly from Excel files.
* **Automated Ranking:** Outputs the final Q, S, and R values and the ranking of alternatives.

## Technologies Used
* Python 3.x
* Pandas (Data manipulation)
* NumPy (Numerical calculations)
* OS (Excel file handling)

## How to Run

1.  **Prepare your Data:**
    * Ensure your dataset starts at **cell A1** in your Excel file.
    * The file format should be `.xlsx`.
2.  **Configuration:**
    * Open the script and enter the correct **target file name** (path) corresponding to your local machine.
3.  **Execution:**
    * Run the script using a Python interpreter or IDLE.
    * The results will be displayed in the console/output file.

---
