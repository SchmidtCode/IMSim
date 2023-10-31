# Inventory Management Simulator

## 👋 Introduction

Welcome to the Inventory Management Simulator, a tool designed to give you hands-on experience managing a virtual inventory. Built using Dash and Python, this simulator allows you to add items, simulate their usage, and track their statuses over time. The simulator aims to replicate the complexities and uncertainties that come with real-world inventory management, making it an invaluable tool for students, professionals, and anyone interested in learning about inventory management.

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

## 📄 Design

**Objective**: To build a simulator using Dash, where each user can manage their own set of items with attributes and observe how these items change over time under various conditions.

**Core Components**:

1. **User Authentication and Data Segregation**:
    - Utilize a simple key-based authentication system where a unique key is generated for each user and saved in their browser’s local storage or cookies. This key serves as the identifier for individual user sessions, allowing for data segregation and personalized simulation experiences without a full-fledged login system.
    - Components: `uuid` for generating unique keys, browser’s local storage or cookies for storing keys.

2. **Data Structure**:
    - Establish a well-structured data schema to represent each user's simulation settings and their inventory of items. Each item should have attributes like usage rate, lead time, cost, etc., while simulation settings should include parameters like simulator speed, review cycle (days), replenishment cost, and more.
    - Utilize dictionaries or custom classes to model user data, settings, and items.

3. **Data Storage**:
    - Utilize local `dcc.Store` to hold the users uuid which will be used by the server to pull the specific user's data from user_data_store

4. **Simulation Logic**:
    - The logic is implemented in Python, triggered by a timer component like `dcc.Interval`, to simulate the evolution of items based on predefined rules and some elements of randomness to mimic real-world scenarios.
    - Components: `dcc.Interval`

5. **User Interface (UI)**:
    - A user-friendly, modern interface created using `dash_bootstrap_components` and `dash_bootstrap_templates` to allow users to interact with their simulation, including starting, stopping, and configuring the simulation.
    - The UI should have a toggle feature to switch between light and dark themes for better user experience.
    - Components: `dash_bootstrap_components`, `dash_bootstrap_templates`

6. **Visualization**:
    - Visual representation of simulation data over time to provide intuitive insights into how item attributes change, utilizing the loaded Bootstrap figure templates for a cohesive look.
    - Components: Plotly, `dash_bootstrap_templates`

7. **Error Handling and Logging**:
    - Implement mechanisms to handle unexpected situations gracefully and log key events and errors for troubleshooting.

8. **Testing and Optimization**:
    - Ensure the app behaves as expected under different scenarios and optimize for performance to handle frequent updates smoothly.

9. **Deployment**:
    - Choose a suitable deployment solution, whether cloud-based or self-hosted, to make the simulator accessible to users.
    - [Plotly Deployment (Heroku)](https://dash.plotly.com/deployment)
    - [The Easiest Way to Deploy Your Dash App for Free](https://towardsdatascience.com/the-easiest-way-to-deploy-your-dash-app-for-free-f92c575bb69e)

**Technical Stack**:

- **Backend**: Primarily Python with Dash for server-side processing.
- **Frontend**: Dash components coupled with Bootstrap for a modern UI, with minimal JavaScript only if necessary.
- **Data Management**: `dcc.Store` for in-app data storage, with the `patch` method to ensure efficient updates.
- **Visualization**: Integrated libraries such as Plotly with Bootstrap templates for real-time data visualization.
- **Authentication**: Simple key-based authentication using unique keys generated and stored in the browser.

**Development Approach**:

1. Initial setup of a Dash app with simple key-based user authentication and Bootstrap integration for modern UI design.
2. Define the data schema and initialize `dcc.Store` with user-specific structures for storing simulation settings and items.
3. Implement simulation logic and tie it to a timer component for regular updates.
4. Build UI components for user interaction, settings configuration, and data visualization, ensuring a cohesive look and the ability to toggle between light and dark themes.
5. Implement error handling and logging mechanisms.
6. Conduct thorough testing to ensure reliability and performance.
7. Deploy the app on a chosen platform for user access.

**Timeline**: Aiming for a basic simulator build within a two-week timeframe, with further optimizations and feature additions as needed.

**Future Considerations**:

- Exploring client-side computations for performance improvements, once comfortable with JavaScript.
- Potentially expanding data storage solutions as the complexity and user base grow.
- Continuous performance monitoring and optimizations to ensure a smooth user experience.
