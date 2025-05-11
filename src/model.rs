use serde::{Serialize, Deserialize};
use ndarray::{Array2, Array1};
use std::path::Path;

#[derive(Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ImageClassifier {
    pub target_width: u32,
    pub target_height: u32,
    
    pub x_train: Array2<f32>,
    pub y_train: Array1<i32>,
    
    pub k: usize,  //KNN number of neighbors
}

impl ImageClassifier {
    pub fn new(target_width: u32, target_height: u32, k: usize) -> Self {
        let num_features = (target_width * target_height) as usize;
        ImageClassifier {
            target_width,
            target_height,
            x_train: Array2::zeros((0, num_features)),
            y_train: Array1::zeros(0),
            k,
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        bincode::serialize_into(file, self)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let model = bincode::deserialize_from(file)?;
        Ok(model)
    }

    // Adicionaremos o método de predição posteriormente
}