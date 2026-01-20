"""
Benchmark Runner - Evaluate LLM performance on planning document GeoJSON extraction

This script:
1. Reads the evaluation dataset from Excel
2. Iterates through each planning document
3. Uses LLM to extract GeoJSON boundaries
4. Compares with ground truth using IoU and other metrics
5. Saves results and model responses for analysis
"""

import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from openrouter_client import OpenRouterClient
from geojson_metrics import (
    load_geojson,
    calculate_spatial_metrics,
)


class BenchmarkRunner:
    """
    Run planning document GeoJSON extraction benchmark across multiple models.
    """

    def __init__(
        self,
        dataset_excel_path: str = "evaluation_data/0_planning_dataset_list.xlsx",
        evaluation_data_dir: str = "evaluation_data",
        results_dir: str = "benchmark_results",
    ):
        """
        Initialize benchmark runner.

        Args:
            dataset_excel_path: Path to Excel file containing dataset info
            evaluation_data_dir: Directory containing evaluation data folders
            results_dir: Directory to save benchmark results
        """
        self.dataset_excel_path = dataset_excel_path
        self.evaluation_data_dir = Path(evaluation_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Load both sheets from the Excel file
        dataset_all = pd.read_excel(
            dataset_excel_path, sheet_name="0_planning_dataset_list"
        )
        dataset_removed = pd.read_excel(
            dataset_excel_path, sheet_name="Shape Mismatch list"
        )

        # Filter out entries whose ID appears in the "Shape Mismatch list" sheet
        # The ~ operator negates the boolean mask, so we keep rows NOT in the mismatch list
        self.dataset_filtered = dataset_all[
            ~dataset_all["Unique ID (Folder_Name)"].isin(
                dataset_removed["Unique ID (Folder_Name)"]
            )
        ]
        print(f"Loaded dataset with {len(self.dataset_filtered)} examples")

    def get_pdf_path(self, folder_name: str) -> Optional[Path]:
        """Find the PDF file in the specified folder."""
        folder_path = self.evaluation_data_dir / folder_name
        if not folder_path.exists():
            return None

        # glob("*.pdf") finds all files matching the pattern in the directory (there is only one PDF file in the folder)
        pdf_files = list(folder_path.glob("*.pdf"))
        return pdf_files[0] if pdf_files else None

    def get_ground_truth_geojson_path(
        self, folder_name, geojson_file_name: str
    ) -> Optional[Path]:
        """Find the ground truth GeoJSON file in the specified folder."""
        folder_path = self.evaluation_data_dir / folder_name
        if not folder_path.exists():
            return None

        geojson_files = list(folder_path.glob(geojson_file_name))
        return geojson_files[0] if geojson_files else None

    def _get_method_dir_name(
        self, method: str, iterative_refinement: bool = False
    ) -> str:
        """
        Get the directory name for a method configuration.

        For linear_transform, appends _iterative or _non_iterative based on the flag.
        """
        if method == "linear_transform":
            suffix = "_iterative" if iterative_refinement else "_non_iterative"
            return f"{method}{suffix}"
        return method

    def save_result(
        self,
        model_name: str,
        example_id: str,
        folder_name: str,
        result_data: Dict[str, Any],
        method: str = "linear_transform",
        iterative_refinement: bool = False,
    ):
        """
        Save benchmark result for a single example.

        Creates folder structure: results_dir/model_name/method_name/folder_name/
        Saves:
        - response.json: Full model response
        - predicted.geojson: Extracted GeoJSON (if successful)
        - metrics.json: Performance metrics
        """
        # Replace "/" with "_" in model names to create valid directory names
        # (e.g., "anthropic/claude-sonnet" becomes "anthropic_claude-sonnet")
        method_dir = self._get_method_dir_name(method, iterative_refinement)
        result_dir = self.results_dir / model_name.replace("/", "_") / method_dir / folder_name
        result_dir.mkdir(parents=True, exist_ok=True)

        response_path = result_dir / "response.json"
        with open(response_path, "w") as f:
            # indent=2 makes the JSON human-readable with 2-space indentation
            json.dump(result_data["full_response"], f, indent=2)

        if result_data.get("predicted_geojson"):
            predicted_path = result_dir / "predicted.geojson"
            with open(predicted_path, "w") as f:
                json.dump(result_data["predicted_geojson"], f, indent=2)

        metrics_path = result_dir / "metrics.json"
        metrics_data = {
            "example_id": example_id,
            "folder_name": folder_name,
            "model": model_name,
            "timestamp": result_data["timestamp"],
            "success": result_data["success"],
            "processing_time": result_data.get("processing_time", 0),
            "tokens": result_data.get("tokens", {}),
            "spatial_metrics": result_data.get("spatial_metrics", {}),
            "error": result_data.get("error"),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        print(f"  Saved results to {result_dir}")

    def run_single_example(
        self,
        client: OpenRouterClient,
        example_row: pd.Series,
        model_name: str,
        method: str = "linear_transform",
        iterative_refinement: bool = False,
    ) -> Dict[str, Any]:
        """
        Run benchmark on a single example.

        Args:
            client: OpenRouterClient instance
            example_row: Row from dataset DataFrame
            model_name: Name of the model being tested
            method: Extraction method ("baseline", "linear_transform", "agentic")
            iterative_refinement: Whether to use iterative refinement (only for linear_transform)

        Returns:
            Result dictionary with metrics and response data
        """
        folder_name = example_row["Unique ID (Folder_Name)"]
        example_id = str(example_row["Sl no"])

        print(f"\nProcessing example {example_id}: {folder_name}")

        pdf_path = self.get_pdf_path(folder_name)

        # Extract just the filename from the full path stored in the Excel column
        geojson_file_name = example_row["geojson ID (for sanity check)"].split("/")[-1]
        gt_geojson_path = self.get_ground_truth_geojson_path(
            folder_name, geojson_file_name
        )

        result = {
            "example_id": example_id,
            "folder_name": folder_name,
            "model": model_name,
            "method": method,
            "iterative_refinement": iterative_refinement,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "full_response": {},
            "spatial_metrics": {},
        }

        if not pdf_path or not pdf_path.exists():
            result["error"] = f"PDF not found in {folder_name}"
            print(f"  Error: {result['error']}")
            return result

        if not gt_geojson_path or not gt_geojson_path.exists():
            result["error"] = f"Ground truth GeoJSON not found in {folder_name}"
            print(f"  Error: {result['error']}")
            return result

        try:
            print(f"  Calling {model_name} to extract GeoJSON (method={method})...")

            # Build kwargs based on method
            extract_kwargs = {"method": method}
            if method == "linear_transform":
                extract_kwargs["iterative_refinement"] = iterative_refinement

            response = client.extract_geojson(
                pdf_path=str(pdf_path),
                **extract_kwargs,
            )

            result["full_response"] = response
            result["processing_time"] = response.get("processing_time", 0)
            result["tokens"] = response.get("tokens", {})

            if not response.get("success"):
                # Use 'error' if it exists, otherwise fall back to 'json_error'
                result["error"] = response.get("error") or response.get("json_error")
                print(f"  Error: {result['error']}")
                return result

            predicted_geojson = response.get("parsed_json")
            if not predicted_geojson:
                result["error"] = "No GeoJSON in response"
                print(f"  Error: {result['error']}")
                return result

            result["predicted_geojson"] = predicted_geojson

            gt_geojson = load_geojson(str(gt_geojson_path))
            if not gt_geojson:
                result["error"] = "Failed to load ground truth GeoJSON"
                print(f"  Error: {result['error']}")
                return result

            print("  Calculating spatial metrics...")
            spatial_metrics = calculate_spatial_metrics(gt_geojson, predicted_geojson)
            result["spatial_metrics"] = spatial_metrics
            result["success"] = True

            print(f"  IoU: {spatial_metrics['iou']:.4f}")
            print(f"  F1 Score: {spatial_metrics['f1_score']:.4f}")
            print(f"  Processing time: {result['processing_time']:.2f}s")

        except Exception as e:
            result["error"] = str(e)
            print(f"  Exception: {e}")

        return result

    def run_benchmark(
        self,
        model_names: List[str],
        method: str = "linear_transform",
        iterative_refinement: bool = False,
        max_examples: Optional[int] = None,
        start_from: int = 0,
    ):
        """
        Run full benchmark across multiple models.

        Args:
            model_names: List of model names/identifiers to test
            method: Extraction method ("baseline", "linear_transform", "agentic")
            iterative_refinement: Whether to use iterative refinement (only for linear_transform)
            max_examples: Maximum number of examples to test (None for all)
            start_from: Index to start from (for resuming)
        """
        method_dir = self._get_method_dir_name(method, iterative_refinement)

        print(f"\n{'=' * 80}")
        print(f"Starting Benchmark Run")
        print(f"{'=' * 80}")
        print(f"Models: {', '.join(model_names)}")
        print(f"Method: {method_dir}")
        print(f"Total examples in dataset: {len(self.dataset_filtered)}")
        if max_examples:
            print(
                f"Testing on: {max_examples} examples (starting from index {start_from})"
            )
        print(f"Results directory: {self.results_dir}")
        print(f"{'=' * 80}\n")

        # iloc[start_from:] selects all rows from index start_from onwards
        dataset_filtered_slice = self.dataset_filtered.iloc[start_from:]
        if max_examples:
            # head(n) returns only the first n rows
            dataset_filtered_slice = dataset_filtered_slice.head(max_examples)

        for model_name in model_names:
            print(f"\n{'=' * 80}")
            print(f"Testing Model: {model_name}")
            print(f"{'=' * 80}")

            try:
                client = OpenRouterClient(model=model_name)
            except Exception as e:
                print(f"Failed to initialize client for {model_name}: {e}")
                continue

            results = []
            successful = 0
            failed = 0

            for idx, row in dataset_filtered_slice.iterrows():
                result = self.run_single_example(
                    client,
                    row,
                    model_name,
                    method=method,
                    iterative_refinement=iterative_refinement,
                )
                results.append(result)

                self.save_result(
                    model_name=model_name,
                    example_id=result["example_id"],
                    folder_name=result["folder_name"],
                    result_data=result,
                    method=method,
                    iterative_refinement=iterative_refinement,
                )

                if result["success"]:
                    successful += 1
                else:
                    failed += 1

                print(
                    f"  Progress: {successful + failed}/{len(dataset_filtered_slice)} "
                    f"(Success: {successful}, Failed: {failed})"
                )

            self._save_model_summary(
                model_name, results, method=method, iterative_refinement=iterative_refinement
            )

            print(f"\n{'=' * 80}")
            print(f"Completed {model_name} ({method_dir})")
            print(
                f"Success: {successful}/{len(dataset_filtered_slice)} ({100 * successful / len(dataset_filtered_slice):.1f}%)"
            )
            print(f"{'=' * 80}\n")

    def _save_model_summary(
        self,
        model_name: str,
        results: List[Dict[str, Any]],
        method: str = "linear_transform",
        iterative_refinement: bool = False,
    ):
        """Save summary statistics for a model and method configuration."""
        method_dir = self._get_method_dir_name(method, iterative_refinement)
        model_method_dir = self.results_dir / model_name.replace("/", "_") / method_dir
        summary_path = model_method_dir / "summary.json"

        # Filter to only successful results for metric calculation
        successful_results = [r for r in results if r["success"]]

        if successful_results:
            ious = np.array([r["spatial_metrics"]["iou"] for r in successful_results])
            precisions = np.array(
                [r["spatial_metrics"]["precision"] for r in successful_results]
            )
            recalls = np.array(
                [r["spatial_metrics"]["recall"] for r in successful_results]
            )
            f1_scores = np.array(
                [r["spatial_metrics"]["f1_score"] for r in successful_results]
            )
            processing_times = np.array(
                [r["processing_time"] for r in successful_results]
            )

            summary = {
                "model": model_name,
                "method": method,
                "iterative_refinement": iterative_refinement if method == "linear_transform" else None,
                "total_examples": len(results),
                "successful": len(successful_results),
                "failed": len(results) - len(successful_results),
                "success_rate": len(successful_results) / len(results),
                "metrics": {
                    "iou": {
                        "mean": ious.mean(),
                        "min": ious.min(),
                        "max": ious.max(),
                        "median": np.median(ious),
                    },
                    "precision": {
                        "mean": np.mean(precisions),
                        "min": np.min(precisions),
                        "max": np.max(precisions),
                        "median": np.median(precisions),
                    },
                    "recall": {
                        "mean": np.mean(recalls),
                        "min": np.min(recalls),
                        "max": np.max(recalls),
                        "median": np.median(recalls),
                    },
                    "f1_score": {
                        "mean": np.mean(f1_scores),
                        "min": np.min(f1_scores),
                        "max": np.max(f1_scores),
                        "median": np.median(f1_scores),
                    },
                    "processing_time": {
                        "mean": np.mean(processing_times),
                        "min": np.min(processing_times),
                        "max": np.max(processing_times),
                        "total": np.sum(processing_times),
                    },
                },
                "timestamp": datetime.now().isoformat(),
            }
        else:
            summary = {
                "model": model_name,
                "method": method,
                "iterative_refinement": iterative_refinement if method == "linear_transform" else None,
                "total_examples": len(results),
                "successful": 0,
                "failed": len(results),
                "success_rate": 0.0,
                "timestamp": datetime.now().isoformat(),
            }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved summary to {summary_path}")


# Example usage
if __name__ == "__main__":
    runner = BenchmarkRunner()
    print(runner.dataset_filtered.head())

    models_to_test = [
        "claude-opus",
        # "gpt-5.2",
        # "gemini-pro",
    ]

    # Available methods: "baseline", "linear_transform", "agentic"
    # For linear_transform, set iterative_refinement=True/False
    runner.run_benchmark(
        model_names=models_to_test,
        method="linear_transform",
        iterative_refinement=True,  # Set to False for linear_transform_non_iterative
        max_examples=10,
        start_from=90,
    )

    print("\n" + "=" * 80)
    print("Benchmark run completed!")
    print(f"Results saved to: {runner.results_dir}")
    print("=" * 80)
