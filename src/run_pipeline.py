"""
Main Pipeline Script for Phase 1: Data Preprocessing and Feature Extraction
Processes MIT-BIH database and creates dataset for XAI tachycardia detection
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json
import pickle
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_loader import MITBIHLoader, ECGRecord
from preprocessing.signal_processing import SignalProcessor, QRSDetector
from preprocessing.beat_segmentation import BeatSegmenter, TachycardiaLabeler
from features.feature_extractor import FeatureExtractor
from features.hrv_features import HRVFeatureExtractor


class TachycardiaDataPipeline:
    """
    Complete data pipeline for tachycardia detection
    
    Steps:
    1. Load MIT-BIH records
    2. Preprocess signals (filtering, normalization)
    3. Segment beats
    4. Extract features
    5. Create labeled dataset
    6. Handle class imbalance
    """
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize pipeline
        
        Args:
            data_dir: Path to mitbih_database folder
            output_dir: Path for processed outputs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'features'), exist_ok=True)
        
        # Initialize components
        self.loader = MITBIHLoader(data_dir)
        self.signal_processor = SignalProcessor(sampling_rate=360)
        self.qrs_detector = QRSDetector(sampling_rate=360)
        self.beat_segmenter = BeatSegmenter(sampling_rate=360)
        self.feature_extractor = FeatureExtractor(sampling_rate=360)
        self.hrv_extractor = HRVFeatureExtractor(sampling_rate=360)
        self.labeler = TachycardiaLabeler(sampling_rate=360)
        
        # Dataset storage
        self.dataset = None
        self.metadata = {}
        
    def run_full_pipeline(self, save_intermediate: bool = True) -> Dict:
        """
        Run complete data processing pipeline
        
        Args:
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary with processed dataset
        """
        print("=" * 60)
        print("XAI TACHYCARDIA DETECTION - DATA PIPELINE")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Records found: {len(self.loader.records)}")
        print()
        
        # Step 1: Process all records
        all_beats = []
        all_features = []
        all_labels_binary = []
        all_labels_multi = []
        all_metadata = []
        all_hrv_features = []
        
        for i, record_id in enumerate(self.loader.records):
            print(f"[{i+1}/{len(self.loader.records)}] Processing record {record_id}...")
            
            try:
                result = self._process_single_record(record_id)
                
                if result['n_beats'] > 0:
                    all_beats.extend(result['beats'])
                    all_features.extend(result['features'])
                    all_labels_binary.extend(result['binary_labels'])
                    all_labels_multi.extend(result['multiclass_labels'])
                    all_metadata.extend(result['metadata'])
                    all_hrv_features.extend(result['hrv_features'])
                    
                    print(f"    Beats: {result['n_beats']}, "
                          f"Tachycardia: {sum(result['binary_labels'])} "
                          f"({100*sum(result['binary_labels'])/result['n_beats']:.1f}%)")
                else:
                    print(f"    No valid beats extracted")
                    
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                continue
        
        # Convert to arrays
        print("\nCreating final dataset...")
        
        self.dataset = {
            'beats': np.array(all_beats),
            'features': np.array(all_features),
            'binary_labels': np.array(all_labels_binary),
            'multiclass_labels': np.array(all_labels_multi),
            'hrv_features': all_hrv_features,
            'metadata': all_metadata,
            'feature_names': self.feature_extractor.get_feature_names()
        }
        
        # Compute statistics
        self._compute_dataset_statistics()
        
        # Save dataset
        if save_intermediate:
            self._save_dataset()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        
        return self.dataset
    
    def _process_single_record(self, record_id: str) -> Dict:
        """Process a single ECG record"""
        
        # Load record
        record = self.loader.load_record(record_id)
        
        # Preprocess both leads
        mlii_processed = self.signal_processor.full_preprocessing(
            record.signal_mlii,
            remove_baseline=True,
            remove_powerline=True,
            remove_hf_noise=True,
            normalize=True
        )
        
        v5_processed = self.signal_processor.full_preprocessing(
            record.signal_v5,
            remove_baseline=True,
            remove_powerline=True,
            remove_hf_noise=True,
            normalize=True
        )
        
        # Segment beats using annotations
        result = self.beat_segmenter.segment_beats_with_labels(
            mlii_processed,
            record.beat_annotations,
            preprocess=False  # Already preprocessed
        )
        
        if result['n_beats'] == 0:
            return {'n_beats': 0, 'beats': [], 'features': [], 
                    'binary_labels': [], 'multiclass_labels': [],
                    'metadata': [], 'hrv_features': []}
        
        # Get R-peak positions
        r_peaks = np.array(result['sample_positions'])
        
        # Compute RR intervals and heart rates
        rr_features = self.beat_segmenter.extract_rr_features(r_peaks)
        
        # Create labels
        binary_labels = self.labeler.label_beats(
            r_peaks,
            record.rhythm_annotations,
            len(record.signal_mlii)
        )
        
        multiclass_labels = self.labeler.create_multiclass_labels(
            r_peaks,
            record.rhythm_annotations
        )
        
        # Extract features for each beat
        features = self.feature_extractor.extract_features_batch(
            result['beats'],
            rr_features
        )
        
        # Extract HRV features (sliding window around each beat)
        hrv_features = []
        rr_intervals = np.diff(r_peaks) / 360  # in seconds
        
        window_size = 20  # beats
        for i in range(len(r_peaks)):
            start = max(0, i - window_size // 2)
            end = min(len(rr_intervals), i + window_size // 2)
            
            if end > start + 5:  # Need at least 5 RR intervals
                window_rr = rr_intervals[start:end]
                hrv = self.hrv_extractor.extract_time_domain_features(window_rr)
            else:
                hrv = self.hrv_extractor._empty_time_domain_features()
            
            hrv_features.append(hrv)
        
        # Create metadata
        metadata = []
        for i in range(len(r_peaks)):
            meta = {
                'record_id': record_id,
                'sample_position': r_peaks[i],
                'beat_type': result['beat_types'][i],
                'time_seconds': r_peaks[i] / 360,
                'heart_rate': rr_features['hr_current'][i],
                'is_tachycardia_by_hr': rr_features['is_tachycardia_hr'][i]
            }
            
            # Get specific tachycardia type if applicable
            if binary_labels[i] == 1:
                meta['tachycardia_type'] = self.labeler.get_tachycardia_type(
                    r_peaks[i], record.rhythm_annotations
                )
            else:
                meta['tachycardia_type'] = 'Normal'
            
            metadata.append(meta)
        
        return {
            'n_beats': result['n_beats'],
            'beats': list(result['beats']),
            'features': list(features),
            'binary_labels': list(binary_labels),
            'multiclass_labels': list(multiclass_labels),
            'metadata': metadata,
            'hrv_features': hrv_features
        }
    
    def _compute_dataset_statistics(self):
        """Compute and store dataset statistics"""
        
        n_total = len(self.dataset['binary_labels'])
        n_tachycardia = sum(self.dataset['binary_labels'])
        n_normal = n_total - n_tachycardia
        
        # Multiclass distribution
        multiclass_counts = {}
        labels = self.dataset['multiclass_labels']
        for label in range(max(labels) + 1):
            multiclass_counts[label] = sum(labels == label)
        
        # Tachycardia type distribution
        tachy_type_counts = {}
        for meta in self.dataset['metadata']:
            t_type = meta['tachycardia_type']
            tachy_type_counts[t_type] = tachy_type_counts.get(t_type, 0) + 1
        
        # Beat type distribution
        beat_type_counts = {}
        for meta in self.dataset['metadata']:
            b_type = meta['beat_type']
            beat_type_counts[b_type] = beat_type_counts.get(b_type, 0) + 1
        
        # Record distribution
        record_counts = {}
        for meta in self.dataset['metadata']:
            r_id = meta['record_id']
            record_counts[r_id] = record_counts.get(r_id, 0) + 1
        
        self.metadata = {
            'total_beats': n_total,
            'tachycardia_beats': n_tachycardia,
            'normal_beats': n_normal,
            'class_imbalance_ratio': n_normal / max(n_tachycardia, 1),
            'multiclass_distribution': multiclass_counts,
            'tachycardia_type_distribution': tachy_type_counts,
            'beat_type_distribution': beat_type_counts,
            'record_distribution': record_counts,
            'n_records': len(self.loader.records),
            'n_features': len(self.dataset['feature_names']),
            'feature_names': self.dataset['feature_names'],
            'beat_length': self.dataset['beats'].shape[1] if n_total > 0 else 0,
            'processing_date': datetime.now().isoformat()
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Total beats: {n_total:,}")
        print(f"Normal beats: {n_normal:,} ({100*n_normal/n_total:.1f}%)")
        print(f"Tachycardia beats: {n_tachycardia:,} ({100*n_tachycardia/n_total:.1f}%)")
        print(f"Class imbalance ratio: {self.metadata['class_imbalance_ratio']:.1f}:1")
        print(f"Number of features: {len(self.dataset['feature_names'])}")
        print(f"Beat window size: {self.metadata['beat_length']} samples")
        
        print("\nTachycardia Type Distribution:")
        for t_type, count in sorted(tachy_type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t_type}: {count:,}")
        
        print("\nMulticlass Distribution:")
        class_names = ['Normal', 'Sinus Tachy', 'SVT', 'VT', 'VFL', 'Other']
        for label, count in sorted(multiclass_counts.items()):
            name = class_names[label] if label < len(class_names) else f'Class_{label}'
            print(f"  {label} ({name}): {count:,}")
    
    def _save_dataset(self):
        """Save processed dataset to disk"""
        
        print("\nSaving dataset...")
        
        # Save as numpy arrays
        np.save(
            os.path.join(self.output_dir, 'processed', 'beats.npy'),
            self.dataset['beats']
        )
        np.save(
            os.path.join(self.output_dir, 'processed', 'features.npy'),
            self.dataset['features']
        )
        np.save(
            os.path.join(self.output_dir, 'processed', 'binary_labels.npy'),
            self.dataset['binary_labels']
        )
        np.save(
            os.path.join(self.output_dir, 'processed', 'multiclass_labels.npy'),
            self.dataset['multiclass_labels']
        )
        
        # Save metadata as JSON
        with open(os.path.join(self.output_dir, 'processed', 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Save beat metadata as CSV
        meta_df = pd.DataFrame(self.dataset['metadata'])
        meta_df.to_csv(
            os.path.join(self.output_dir, 'processed', 'beat_metadata.csv'),
            index=False
        )
        
        # Save HRV features
        hrv_df = pd.DataFrame(self.dataset['hrv_features'])
        hrv_df.to_csv(
            os.path.join(self.output_dir, 'features', 'hrv_features.csv'),
            index=False
        )
        
        # Save feature names
        with open(os.path.join(self.output_dir, 'features', 'feature_names.json'), 'w') as f:
            json.dump(self.dataset['feature_names'], f, indent=2)
        
        # Save complete dataset as pickle for easy loading
        with open(os.path.join(self.output_dir, 'processed', 'complete_dataset.pkl'), 'wb') as f:
            pickle.dump({
                'beats': self.dataset['beats'],
                'features': self.dataset['features'],
                'binary_labels': self.dataset['binary_labels'],
                'multiclass_labels': self.dataset['multiclass_labels'],
                'feature_names': self.dataset['feature_names'],
                'metadata': self.metadata
            }, f)
        
        print(f"Dataset saved to {self.output_dir}")
    
    def load_dataset(self) -> Dict:
        """Load previously saved dataset"""
        
        dataset_path = os.path.join(self.output_dir, 'processed', 'complete_dataset.pkl')
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run pipeline first.")
        
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def get_train_test_split(self, test_size: float = 0.2,
                              stratify: bool = True,
                              random_state: int = 42) -> Dict:
        """
        Create train/test split (patient-wise to avoid data leakage)
        
        Args:
            test_size: Fraction for test set
            stratify: Whether to stratify by class
            random_state: Random seed
            
        Returns:
            Dictionary with train/test splits
        """
        from sklearn.model_selection import train_test_split
        
        if self.dataset is None:
            self.dataset = self.load_dataset()
        
        # Get unique records
        meta_df = pd.DataFrame(self.dataset['metadata'])
        records = meta_df['record_id'].unique()
        
        # Split by records (not by beats) to avoid data leakage
        np.random.seed(random_state)
        np.random.shuffle(records)
        
        n_test = int(len(records) * test_size)
        test_records = set(records[:n_test])
        train_records = set(records[n_test:])
        
        # Create masks
        train_mask = meta_df['record_id'].isin(train_records).values
        test_mask = meta_df['record_id'].isin(test_records).values
        
        split = {
            'X_train': self.dataset['features'][train_mask],
            'X_test': self.dataset['features'][test_mask],
            'y_train_binary': self.dataset['binary_labels'][train_mask],
            'y_test_binary': self.dataset['binary_labels'][test_mask],
            'y_train_multi': self.dataset['multiclass_labels'][train_mask],
            'y_test_multi': self.dataset['multiclass_labels'][test_mask],
            'beats_train': self.dataset['beats'][train_mask],
            'beats_test': self.dataset['beats'][test_mask],
            'train_records': list(train_records),
            'test_records': list(test_records)
        }
        
        print(f"Train set: {sum(train_mask)} beats from {len(train_records)} records")
        print(f"Test set: {sum(test_mask)} beats from {len(test_records)} records")
        print(f"Train tachycardia: {sum(split['y_train_binary'])} ({100*sum(split['y_train_binary'])/len(split['y_train_binary']):.1f}%)")
        print(f"Test tachycardia: {sum(split['y_test_binary'])} ({100*sum(split['y_test_binary'])/len(split['y_test_binary']):.1f}%)")
        
        return split


def main():
    """Run the data pipeline"""
    
    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_dir, 'mitbih_database')
    output_dir = os.path.join(project_dir, 'data')
    
    # Check data directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please ensure the mitbih_database folder is in the project directory.")
        sys.exit(1)
    
    # Run pipeline
    pipeline = TachycardiaDataPipeline(data_dir, output_dir)
    dataset = pipeline.run_full_pipeline(save_intermediate=True)
    
    # Create train/test split
    print("\nCreating train/test split...")
    split = pipeline.get_train_test_split(test_size=0.2)
    
    # Save split
    np.savez(
        os.path.join(output_dir, 'processed', 'train_test_split.npz'),
        **split
    )
    
    print("\nPipeline complete!")
    print(f"Dataset saved to: {output_dir}")
    
    return dataset, split


if __name__ == '__main__':
    main()
