// src/predict.rs
use crate::model::ImageClassifier;
use image::DynamicImage;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::Distances;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use std::path::Path;

#[derive(Debug)]
pub enum PredictError {
    ModelLoad(Box<dyn std::error::Error>),
    ImageProcessing(String),
    PredictionError(String),
}

impl std::fmt::Display for PredictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelLoad(e) => write!(f, "Model load error: {}", e),
            Self::ImageProcessing(msg) => write!(f, "Image processing error: {}", msg),
            Self::PredictionError(msg) => write!(f, "Prediction error: {}", msg),
        }
    }
}

impl std::error::Error for PredictError {}

fn preprocess_image(img: &DynamicImage, width: u32, height: u32) -> Vec<f32> {
    let gray_img = img.to_luma8();
    let resized = image::imageops::resize(
        &gray_img,
        width,
        height,
        image::imageops::FilterType::Triangle,
    );
    resized.iter().map(|&p| p as f32 / 255.0).collect()
}

pub fn classify_image(model_path: &Path, image_path: &Path) -> Result<i32, PredictError> {
    let model = ImageClassifier::load(model_path).map_err(PredictError::ModelLoad)?;

    let img = image::open(image_path)
        .map_err(|e| PredictError::ImageProcessing(format!("Failed to open image: {}", e)))?;

    let features = preprocess_image(&img, model.target_width, model.target_height);

    let sample = DenseMatrix::new(1, features.len(), features, false)
        .map_err(|e| PredictError::PredictionError(format!("Matrix creation error: {}", e)))?;

    // Convert in DataTrain
    let x_train = DenseMatrix::new(
        model.x_train.nrows(),
        model.x_train.ncols(),
        model.x_train.iter().cloned().collect(),
        false,
    )
    .map_err(|e| PredictError::PredictionError(format!("Matrix creation error: {}", e)))?;

    let y_train: Vec<i32> = model.y_train.iter().cloned().collect();

    // Configurar parâmetros do KNN
    let params = KNNClassifierParameters::default()
        .with_k(model.k)
        .with_distance(Distances::euclidian());

    let knn = KNNClassifier::fit(&x_train, &y_train, params)
        .map_err(|e| PredictError::PredictionError(format!("Fit error: {}", e)))?;

    let prediction = knn
        .predict(&sample)
        .map_err(|e| PredictError::PredictionError(format!("Prediction error: {}", e)))?;

    Ok(prediction[0])
}
