# IMSim (Inventory Management Simulator)

## 👋 Introduction

Welcome to the Inventory Management Simulator, a tool designed to give you hands-on experience managing a virtual inventory. This Simulator is designed to show how viewing items' PNA in terms of days from OP can be beneficial.

This siumulator is built using Dash and Python. The IMSim allows you to add items, simulate their usage, and track their statuses over time. The simulator aims to replicate the complexities and uncertainties that come with real-world inventory management, making it an invaluable tool for students, professionals, and anyone interested in learning about inventory management.

## 🚀 Getting started with this simulator

### Requirements

1. **Python**: Make sure you have Python 3.11.5 installed.
2. **Anaconda** (Optional): If you prefer using Anaconda, make sure you have it installed.

### Installation and Running

#### Using `pip`

1. **Clone the Repository**: Run `git clone https://github.com/SchmidtCode/IMSim` to clone the project repository to your local machine.
2. **Navigate to Project Directory**: Use `cd <directory_name>` to navigate into the root directory of the project.
3. **Install Dependencies**: Run `pip install -r requirements.txt` to install the required packages.
4. **Run the App**: Use `python app.py` to start the Dash app. The app will be accessible at `http://127.0.0.1:8050/` by default.

#### Using Anaconda

1. **Clone the Repository**: Run `git clone https://github.com/SchmidtCode/IMSim` to clone the project repository to your local machine.
2. **Navigate to Project Directory**: Use `cd <directory_name>` to navigate into the root directory of the project.
3. **Create a New Environment**: Run `conda env create -f environment.yml` to create a new Anaconda environment with the required packages.
4. **Activate the Environment**: Use `conda activate <env_name>` to activate the new environment.
5. **Run the App**: Use `python app.py` to start the Dash app. The app will be accessible at `http://127.0.0.1:8050/` by default.

### Basic Usage

1. **Initialize Settings**: After launching the app, you will find options to set initial parameters such as simulator speed, review cycle, and others.
2. **Add Items**: Navigate to the "Items" tab and add items you want to manage. Here you can specify the item's attributes like usage rate, lead time, cost, etc.
3. **Start Simulation**: Once the settings and items are initialized, navigate to the "Simulate" tab and click "Start Simulation".
