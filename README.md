# Skin Diagnosis Application

A web application for skin lesion diagnosis using SwinTransformer deep learning model.

## Setup Instructions

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Make sure the model file is in the project directory:**
   - The model file `SwinTransformer_best.pth` should be in the project root directory.

3. **Run the Flask server:**
   ```bash
   python app.py
   ```

4. **Access the application:**
   - Open your browser and navigate to `http://localhost:5000`
   - The server will run on port 5000 by default

## Features

- **User Authentication:**
  - Sign up with email and password
  - Login with registered credentials
  - Session management

- **Image Analysis:**
  - Upload skin lesion images
  - Get predictions for 7 different lesion types
  - View probability scores for all conditions

- **Results Display:**
  - Shows top 1-2 predictions based on probability
  - Displays confirmation tests for predicted conditions
  - Provides consultation recommendations

## Usage

1. **Sign Up:** Create a new account with your email and password
2. **Login:** Use your credentials to access the application
3. **Upload Image:** Go to the input page and upload a skin lesion image
4. **Analyze:** Click the "Analyze Image" button to get predictions
5. **View Results:** See the diagnosis results with probabilities and confirmation tests

## API Endpoints

- `POST /api/signup` - Create a new user account
- `POST /api/login` - Login with email and password
- `POST /api/logout` - Logout current user
- `GET /api/user` - Get current user information
- `POST /api/predict` - Predict skin lesion type from uploaded image

## Model Information

The application uses a SwinTransformer model (`swin_base_patch4_window7_224`) trained for 7-class skin lesion classification:
- Actinic Keratoses / Intraepithelial Carcinoma (akiec)
- Basal Cell Carcinoma (bcc)
- Benign Keratosis-like Lesions (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic Nevi (nv)
- Vascular Lesions (vasc)

