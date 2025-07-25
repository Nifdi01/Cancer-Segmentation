import pandas as pd
from collections import defaultdict
from .Evaluate import predict_and_evaluate_grid_sizes


def analyze_iou_across_images(output_dir, model, index_range, mask_threshold=0.7, grid_range=(4, 10)):
    """
    Analyze IoU scores across multiple images and return comprehensive results.

    Args:
        output_dir: Directory containing cropped images
        model: Trained segmentation model
        mask_threshold: Threshold for binarizing predictions
        grid_range: Tuple of (min_grid_size, max_grid_size) inclusive
        index_range: List of randomly generated indices for sampling the image set

    Returns:
        results: Dictionary containing best IoU scores, best grid sizes, and all scores
    """
    best_iou_scores = []  # Best IoU for each image
    best_grid_sizes = []  # Best grid size for each image
    all_iou_scores = defaultdict(list)  # All IoU scores by grid size
    failed_images = []  # Images that failed processing
    end_idx = len(index_range)+1
    counter = 1

    for image_idx in index_range:
        print(f"Processing image {counter}/{end_idx-1}...", end=" ")

        try:
            best_grid_size, iou_scores = predict_and_evaluate_grid_sizes(
                output_dir=output_dir,
                image_idx=image_idx,
                model=model,
                mask_threshold=mask_threshold,
                grid_range=grid_range
            )

            if best_grid_size is not None and iou_scores:
                # Store the best IoU score for this image
                best_iou = iou_scores[best_grid_size]
                best_iou_scores.append(best_iou)
                best_grid_sizes.append(best_grid_size)

                # Store all IoU scores for this image by grid size
                for grid_size, iou in iou_scores.items():
                    all_iou_scores[grid_size].append(iou)

                print(f"✓ Best: {best_grid_size}x{best_grid_size} (IoU: {best_iou:.4f})")
            else:
                print("✗ Failed")
                failed_images.append(image_idx)

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            failed_images.append(image_idx)

        counter += 1

    results = {
        'best_iou_scores': best_iou_scores,
        'best_grid_sizes': best_grid_sizes,
        'all_iou_scores': dict(all_iou_scores),
        'failed_images': failed_images,
        'processed_images': len(best_iou_scores),
        'total_images': len(index_range)
    }

    return results


def get_summary(results):
    """
    Create a summary DataFrame with statistics by grid size.

    Args:
        results: Dictionary from analyze_iou_across_images function

    Returns:
        df: pandas DataFrame with summary statistics
    """

    # Summary statistics by grid size
    summary_records = []

    for grid_size, iou_scores in results['all_iou_scores'].items():
        summary_records.append({
            'grid_size': grid_size,
            'mean_iou': pd.Series(iou_scores).mean(),
            'std_iou': pd.Series(iou_scores).std(),
            'min_iou': pd.Series(iou_scores).min(),
            'max_iou': pd.Series(iou_scores).max(),
            'median_iou': pd.Series(iou_scores).median(),
        })

    df_summary = pd.DataFrame(summary_records)
    return df_summary

