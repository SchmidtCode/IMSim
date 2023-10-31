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

1. **Clone the Repository**: Run `git clone <repo_url>` to clone the project repository to your local machine.
2. **Navigate to Project Directory**: Use `cd <directory_name>` to navigate into the root directory of the project.
3. **Install Dependencies**: Run `pip install -r requirements.txt` to install the required packages.
4. **Run the App**: Use `python app.py` to start the Dash app. The app will be accessible at `http://127.0.0.1:8050/` by default.

#### Using Anaconda

1. **Clone the Repository**: Run `git clone <repo_url>` to clone the project repository to your local machine.
2. **Navigate to Project Directory**: Use `cd <directory_name>` to navigate into the root directory of the project.
3. **Create a New Environment**: Run `conda env create -f environment.yml` to create a new Anaconda environment with the required packages.
4. **Activate the Environment**: Use `conda activate <env_name>` to activate the new environment.
5. **Run the App**: Use `python app.py` to start the Dash app. The app will be accessible at `http://127.0.0.1:8050/` by default.

### Basic Usage

1. **Initialize Settings**: After launching the app, you will find options to set initial parameters such as simulator speed, review cycle, and others.
2. **Add Items**: Navigate to the "Items" tab and add items you want to manage. Here you can specify the item's attributes like usage rate, lead time, cost, etc.
3. **Start Simulation**: Once the settings and items are initialized, navigate to the "Simulate" tab and click "Start Simulation".

## ✅ Todo

- [x] Refactor Code to have users store sim settings and items
  - [x] Move User Data to server but keep UUID local to user
    - [x] Add Persistence to user data on the server
  - [ ] Ensure what user stores and passes between server is minimized to limit performance impact
- [x] Build custom order modal (Why is graph not updating after custom order?)
  - [ ] Add helpful metrics to custom order screen
    - [ ] Cost & Extension?
    - [ ] Available On Hand, On Order, & Backorder
    - [ ] Detailed Usage over time
- [x] Add reality (normalized randomness) to simulator, may want to include hits/month to item variables
  - [x] Add hits ratings to items based on CSD
  - [x] Use new hits to randomize usage (did pna decrease that day (hits), then by how much (usage and hits))
- [ ] Add item import module
- [ ] Add End of Month item recalculaiton based on CSD
  - [ ] log monthly usage, hits and lead time
  - [ ] recalculate item usage, hits and lead time at EOM based on CSD
- [ ] Add On Hand tracking
  - [ ] Store Order Data (Lead Time for when it arrives)
  - [ ] Track On Hand Quantity
  - [ ] Create graph of On Hand Quantity
- [ ] Add item manager module
- [ ] Add manual mins/keeps based on CSD
- [ ] Add warehouse storage and pallet quantity
- [ ] Build deployment tutorial
- [ ] Add Dark Mode toggle
