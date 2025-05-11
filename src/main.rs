// src/main.rs
mod model;
mod train;
mod predict;

use clap::Parser;
use std::{arch::x86_64, path::PathBuf};

#[derive(Parser)]
#[command(name = "Image Classifier")]
enum Command {
    #[command(name = "train")]
    Train {
        #[arg(long)]
        good_images: PathBuf,
        
        #[arg(long)]
        bad_images: PathBuf,
        
        #[arg(long, default_value = "model.bin")]
        output: PathBuf,
        
        #[arg(long, default_value_t = 64)]
        width: u32,
        
        #[arg(long, default_value_t = 64)]
        height: u32,
        
        #[arg(long, default_value_t = 3)]
        k: usize,
    },

    #[command(name = "classify")]
    Classify {
        #[arg(long)]
        model: PathBuf,
        
        #[arg(long)]
        image: PathBuf,
    },
    
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    env_logger::init();

    let args = Command::parse();
    
    match args {
        Command::Train { good_images, bad_images, output, width, height, k } => {
            train::train_model(
                &good_images,
                &bad_images,
                &output,
                width,
                height,
                k,
            )?;
            println!("Model trained and saved to: {:?}", output);
        }

        Command::Classify { model, image } => {
            let prediction = predict::classify_image(&model, &image)?;
            let class_name = if prediction == 1 { "GOOD" } else { "BAD" };
            println!("Image classified as: {} (confidence: {})", class_name, prediction); //confidence not implemented yet
        }
    }
    
    Ok(())
}