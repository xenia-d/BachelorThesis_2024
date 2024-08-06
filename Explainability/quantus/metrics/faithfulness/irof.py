"""This module contains the implementation of the Iterative Removal of Features metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import auc

from quantus.functions.perturb_func import baseline_replacement_by_indices
import matplotlib.pyplot as plt
from quantus.helpers import asserts, utils, warn
from quantus.helpers.enums import (
    DataType,
    EvaluationCategory,
    ModelType,
    ScoreDirection,
)
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.perturbation_utils import make_perturb_func
from quantus.metrics.base import Metric

if sys.version_info >= (3, 8):
    from typing import final
else:
    from typing_extensions import final


@final
class IROF(Metric[List[float]]):
    """
    Implementation of IROF (Iterative Removal of Features) by Rieger at el., 2020.

    The metric computes the area over the curve per class for sorted mean importances
    of feature segments (superpixels) as they are iteratively removed (and prediction scores are collected),
    averaged over several test samples.

    Assumptions:
        - The original metric definition relies on image-segmentation functionality. Therefore, only apply the
        metric to 3-dimensional (image) data. To extend the applicablity to other data domains,
        adjustments to the current implementation might be necessary.

    References:
        1) Laura Rieger and Lars Kai Hansen. "Irof: a low resource evaluation metric for
        explanation methods." arXiv preprint arXiv:2003.08747 (2020).

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "IROF"
    data_applicability = {DataType.IMAGE}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.HIGHER
    evaluation_category = EvaluationCategory.FAITHFULNESS

    def __init__(
        self,
        task: str,
        segmentation_method: str = "slic",
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Optional[Callable] = None,
        perturb_baseline: str = "mean",
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = True,
        aggregate_func: Optional[Callable] = None,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        segmentation_method: string
            Image segmentation method:'slic' or 'felzenszwalb', default="slic".
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func: callable
            Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline: string
            Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="mean".
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        kwargs: optional
            Keyword arguments.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )
        self.task = task

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices

        # Save metric-specific attributes.
        self.segmentation_method = segmentation_method
        self.nr_channels = None
        self.perturb_func = make_perturb_func(
            perturb_func, perturb_func_kwargs, perturb_baseline=perturb_baseline
        )

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self._class.name_,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and the method to segment "
                    "the image 'segmentation_method' (including all its associated"
                    " hyperparameters), also, IROF only works with image data"
                ),
                data_domain_applicability=(
                    f"Also, the current implementation only works for 3-dimensional (image) data."
                ),
                citation=(
                    "Rieger, Laura, and Lars Kai Hansen. 'Irof: a low resource evaluation metric "
                    "for explanation methods.' arXiv preprint arXiv:2003.08747 (2020)"
                ),
            )

    def __call__(
        self,
        model,
        task: str,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = True,
        device: Optional[str] = None,
        batch_size: int = 500,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to evaluation_scores.
        Calls custom_postprocess() afterwards. Finally returns evaluation_scores.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func: callable
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this _call_ only and won't be saved as attribute. If None, self.softmax is used.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        kwargs: optional
            Keyword arguments.

        Returns
        -------
        evaluation_scores: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >> import quantus
            >> from quantus import LeNet
            >> import torch

            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Metric(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency)
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=None,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            batch_size=batch_size,
            **kwargs,
        )



    def move_tensor_to_device(tensor, device):
        return tensor.to(device)

    def evaluate_instance_segmentation(
        self,
        model: ModelInterface,
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
    ) -> float:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x: np.ndarray
            The input to be evaluated on an instance-basis.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.
        Returns
        -------
        float
            The evaluation results.
        """
        
        # Set the device to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        model.to(device)
        
        
        # Ensure input tensor is on the appropriate device
        x_input = model.shape_input(x, x.shape, channel_first=True)
        x_input_tensor = torch.from_numpy(x_input).float().to(device)
    

        if hasattr(model, 'model'):
            output = model.model(x_input_tensor)
        else:
            output = model(x_input_tensor)
        
        y_pred = torch.exp(output[0]).detach().cpu().numpy()

   
        class_index = 0 # Select class of interest here (0-6 for Cityscapes)
        mask = (y == class_index)

        y_pred_squeezed = np.squeeze(y_pred, axis=0)

        # Get the scores for target pixels
        scores = y_pred_squeezed[class_index][mask]
        # Average scores for target pixels
        average_y_pred_score = np.mean(scores)


        print("average score original", average_y_pred_score)


        # Segment image.
        segments = utils.get_superpixel_segments(
            img=np.moveaxis(x, 0, -1).astype("double"),
            segmentation_method=self.segmentation_method,
        )
        nr_segments = len(np.unique(segments))
        asserts.assert_nr_segments(nr_segments=nr_segments)

        # Calculate average attribution of each segment.
        att_segs = np.zeros(nr_segments)
        for i, s in enumerate(range(nr_segments)):
            att_segs[i] = np.mean(a[:, segments == s])

        # Sort segments based on the mean attribution (descending order).
        s_indices = np.argsort(-att_segs)

        preds = []
        x_prev_perturbed = x

        for i_ix, s_ix in enumerate(s_indices):
            # Perturb input by indices of attributions.
            a_ix = np.nonzero((segments == s_ix).flatten())[0]

            x_perturbed = self.perturb_func(
                arr=x_prev_perturbed,
                indices=a_ix,
                indexed_axes=self.a_axes,
            )
            warn.warn_perturbation_caused_no_change(
                x=x_prev_perturbed, x_perturbed=x_perturbed
            )

            # Predict on perturbed input x.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            
            # Convert input to tensor and move to appropriate device
            x_input_tensor = torch.from_numpy(x_input).float().to(model.device)
            
            # Move input tensor to GPU
            x_input_tensor = x_input_tensor.to('cuda')
            
            # Move model to GPU
            model.to('cuda')

            
            # Ensure model output is handled correctly
            if hasattr(model, 'model'):
                output = model.model(x_input_tensor)
            else:
                output = model(x_input_tensor)
        

            y_pred_perturb = torch.exp(output[0]).detach().cpu().numpy()
            
            mask = (y == class_index)

           
            y_pred_perturb_squeezed = np.squeeze(y_pred_perturb, axis=0)

            # Get the scores for target pixels
            scores = y_pred_perturb_squeezed[class_index][mask]

            # Average scores for target pixels
            average_y_pred_perturb_score = np.mean(scores)

            print("average perturbed score", average_y_pred_perturb_score)
            
            # Normalize the scores to be within range [0, 1].
            average_ratio = np.mean(average_y_pred_perturb_score / average_y_pred_score)
            preds.append(float(average_ratio))
            x_prev_perturbed = x_perturbed


        # # Plot AOC curve
        # plt.figure()
        # plt.plot(range(len(preds)), preds, marker='o')
        # plt.title('AOC Curve')
        # plt.xlabel('Number of Segments Removed')
        # plt.ylabel('Class Score')
        # plt.grid(True)
        # plt.show()


        aoc = len(preds) - utils.calculate_auc(np.array(preds))
        
        if not np.isnan(aoc):
            print("AOC value: ", aoc, "class: ", class_index)
            return aoc
        else:
            return 0  # Neutral value


    
    def evaluate_instance_depth(
        self,
        model: ModelInterface,
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
    ) -> float:
        """
        Evaluate instance for the depth task.

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x: np.ndarray
            The input to be evaluated on an instance-basis.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.

        Returns
        -------
        float
            The evaluation results.
        """

        # Set the device to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        
        # Ensure input tensor is on the appropriate device
        x_input = model.shape_input(x, x.shape, channel_first=True)
        x_input_tensor = torch.from_numpy(x_input).float().to(device)
    
        # Ensure model output is handled correctly
        if hasattr(model, 'model'):
            output = model.model(x_input_tensor)
        else:
            output = model(x_input_tensor)
    

        # Move input tensor to appropriate device (e.g., GPU)
        if torch.cuda.is_available():
            x_input_tensor = x_input_tensor.to('cuda')
        
        # Ensure model parameters are on the same device as the input tensor
        model = model.to(x_input_tensor.device)

        # Ensure model output is handled correctly
        if hasattr(model, 'model'):
            output = model.model(x_input_tensor)
        else:
            output = model(x_input_tensor)
        

        y_pred = torch.exp(output[1]).detach().cpu().numpy()

        # Segment image.
        segments = utils.get_superpixel_segments(
            img=np.moveaxis(x, 0, -1).astype("double"),
            segmentation_method=self.segmentation_method,
        )
        nr_segments = len(np.unique(segments))
        asserts.assert_nr_segments(nr_segments=nr_segments)

        # Calculate average attribution of each segment.
        att_segs = np.zeros(nr_segments)
        for i, s in enumerate(range(nr_segments)):
            att_segs[i] = np.mean(a[:, segments == s])

        # Sort segments based on the mean attribution (descending order).
        s_indices = np.argsort(-att_segs)

        variations = []
        x_prev_perturbed = x

        for i_ix, s_ix in enumerate(s_indices):
            # Perturb input by indices of attributions.
            a_ix = np.nonzero((segments == s_ix).flatten())[0]

            x_perturbed = self.perturb_func(
                arr=x_prev_perturbed,
                indices=a_ix,
                indexed_axes=self.a_axes,
            )
            warn.warn_perturbation_caused_no_change(
                x=x_prev_perturbed, x_perturbed=x_perturbed
            )

            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            
            # Convert input to tensor and move to appropriate device
            x_input_tensor = torch.from_numpy(x_input).float().to(model.device)
            
            # Move input tensor to GPU
            x_input_tensor = x_input_tensor.to('cuda')
            
            # Move model to GPU
            model.to('cuda')
            
            
            # Ensure model output is handled correctly
            if hasattr(model, 'model'):
                output = model.model(x_input_tensor)
            else:
                output = model(x_input_tensor)
            

            y_pred_perturb = torch.exp(output[1]).detach().cpu().numpy()


            # # Plot depth map after perturbation
            # plt.figure()
            # plt.imshow(y_pred_perturb_2d, cmap='viridis')  # Assuming 'jet' colormap for depth visualization
            # plt.title('Depth Map after Iteration {}'.format(i_ix + 1))
            # plt.colorbar(label='Depth')
            # plt.axis('off')
            # plt.show()

            # variation = abs((average_y_pred_perturb - average_y_pred) / average_y_pred) * 100

            # Calculate per-pixel absolute difference between original and perturbed depth maps
            pixel_diff = np.abs((y_pred_perturb - y_pred)/y_pred)
            # Calculate the average variation
            variation = np.mean(pixel_diff) * 100


            print("%variation after perturbation", variation)
            variations.append(variation)

            # # Create a new figure with subplots
            # plt.figure(figsize=(12, 6))

            # # Plot depth map after perturbation
            # plt.subplot(1, 2, 1)  # Subplot with 1 row, 2 columns, and index 1
            # plt.imshow(y_pred_perturb_2d, cmap='viridis')  # Assuming 'viridis' colormap for depth visualization
            # plt.title('Depth Map after Iteration {}'.format(i_ix + 1))
            # plt.colorbar(label='Depth')
            # plt.axis('off')

            # # # Plot perturbed image
            # plt.subplot(1, 2, 2)  # Subplot with 1 row, 2 columns, and index 2
            # plt.imshow(np.transpose(x_perturbed, (1, 2, 0)))
            # plt.title('Perturbed Image after Iteration {}'.format(i_ix + 1))
            # plt.axis('off')

            # # Adjust layout
            # plt.tight_layout()

            # # Show the plots
            # plt.show()
            x_prev_perturbed = x_perturbed


        # Plot AUC curve
        plt.figure()
        plt.plot(range(len(variations)), variations, marker='o')
        plt.title('IROF Curve')
        plt.xlabel('Number of Segments Removed')
        plt.ylabel('Absolute % Variation')
        plt.grid(True)
        plt.show()

        # Calculate AUC
        auc_value = auc(range(len(variations)), variations)
        return auc_value


    def custom_preprocess(
        self,
        x_batch: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Implementation of custom_preprocess_batch.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        custom_batch: any
            Gives flexibility ot the user to use for evaluation, can hold any variable.

        Returns
        -------
        None
        """
        # Infer number of input channels.
        self.nr_channels = x_batch.shape[1]

    @property
    def get_aoc_score(self):
        """Calculate the area over the curve (AOC) score for several test samples."""
        return np.mean(self.evaluation_scores)

    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        **kwargs,
    ) -> List[float]:
        """
        This method performs XAI evaluation on a single batch of explanations.
        For more information on the specific logic, we refer the metric’s initialisation docstring.

        Parameters
        ----------
        model: ModelInterface
            A ModelInterface that is subject to explanation.
        x_batch: np.ndarray
            The input to be evaluated on a batch-basis.
        y_batch: np.ndarray
            The output to be evaluated on a batch-basis.
        a_batch: np.ndarray
            The explanation to be evaluated on a batch-basis.
        kwargs:
            Unused.

        Returns
        -------
        scores_batch:
            The evaluation results.
        """
        
        all_irof_scores = [] 
    
        # Iterate over each instance in the batch and compute IROF score for each
        for x, y, a in zip(x_batch, y_batch, a_batch):
            if self.task == "segmentation":
                irof_score = self.evaluate_instance_segmentation(model=model, x=x, y=y, a=a)
                all_irof_scores.append(irof_score)  # Append the computed IROF score to the list
            elif self.task == "depth":
                irof_score = self.evaluate_instance_depth(model=model, x=x, y=y, a=a)
                all_irof_scores.append(irof_score)  # Append the computed IROF score to the list
            else:
                raise ValueError("Invalid task provided.")
    
            print("testing the list", all_irof_scores)
    
        return all_irof_scores  # Return the list containing all IROF scores
                