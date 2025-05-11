use crate::model::ImageClassifier;
use image::{imageops::FilterType, DynamicImage};
use std::path::{Path, PathBuf};

//Enum Errors
#[derive(Debug)]
pub enum TrainError {
    ImageProcessing(String),
    ModelSave(std::io::Error),
}

impl std::fmt::Display for TrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImageProcessing(msg) => write!(f, "Image processing error: {}", msg),
            Self::ModelSave(e) => write!(f, "Model save error: {}", e),
        }
    }
}

impl std::error::Error for TrainError {}

fn process_directory(
    dir_path: &Path,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<Vec<f32>>, TrainError> {

    

    let mut features = Vec::new();
    
    for entry in std::fs::read_dir(dir_path).map_err(|e| {
        TrainError::ImageProcessing(format!("Failed to read directory: {}", e))
    })? {

        
        let path = entry.map_err(|e| {
            TrainError::ImageProcessing(format!("Invalid directory entry: {}", e))
        })?.path();

        log::info!("Processing directory: {:?}", dir_path);
    
        if path.is_file() {
            let img = image::open(&path).map_err(|e| {
                TrainError::ImageProcessing(format!("Failed to open image: {}", e))
            })?;

            if !path.is_file() || !is_supported_image_format(&path) {
                continue;
            }

            log::debug!("Processing image: {:?}", path);
            
            let processed = preprocess_image(&img, target_width, target_height);
            features.push(processed);
        }
    }
    
    Ok(features)
}

fn is_supported_image_format(path: &Path) -> bool {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png")
}

// Image Pre-Processing
fn preprocess_image(img: &image::DynamicImage, width: u32, height: u32) -> Vec<f32> {
    // Converter para escala de cinza
    let gray_img = img.to_luma8();
    
    // Redimensionar
    let resized = image::imageops::resize(
        &gray_img,
        width,
        height,
        FilterType::Triangle, // Filter can be changed to other types, but change the performance
    );
    
    // Normalizar pixels para 0-1
    resized.iter()
        .map(|&p| p as f32 / 255.0)
        .collect()
}

// Função principal de treinamento
pub fn train_model(
    good_path: &Path,
    bad_path: &Path,
    model_path: &Path,
    target_width: u32,
    target_height: u32,
    k: usize,
) -> Result<(), TrainError> {
    
    //Good images vec
    let good_features = process_directory(good_path, target_width, target_height)?;
    
    // Bad images vec
    let bad_features = process_directory(bad_path, target_width, target_height)?;
    
    // train arrays
    let num_features = (target_width * target_height) as usize;
    let mut x_train = Vec::with_capacity(good_features.len() + bad_features.len());
    let mut y_train = Vec::with_capacity(good_features.len() + bad_features.len());
    
    
    for feat in good_features {
        x_train.push(feat);
        y_train.push(1); // Class "good"
    }
    
    for feat in bad_features {
        x_train.push(feat);
        y_train.push(0); // Class "bad"
    }
    
    let x_array = ndarray::Array2::from_shape_vec(
        (x_train.len(), num_features),
        x_train.into_iter().flatten().collect()
    ).unwrap();
    
    let y_array = ndarray::Array1::from(y_train);
    
    // Criar e salvar o modelo
    let model = ImageClassifier {
        target_width,
        target_height,
        x_train: x_array,
        y_train: y_array,
        k,
    };
    
    model.save(model_path).unwrap(); // Change the unwrap to handle errors properly (future)

    
    Ok(())
}