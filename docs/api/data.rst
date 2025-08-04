Data Utilities
==============

The data utilities module provides dataset classes and preprocessing functions.

.. currentmodule:: wimae.training.data_utils

Dataset Classes
---------------

.. autoclass:: OptimizedPreloadedDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ScenarioSplitDataset
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing Functions
-----------------------

.. autofunction:: calculate_complex_statistics

.. autofunction:: normalize_complex_matrix

.. autofunction:: denormalize_complex_matrix

.. autofunction:: create_efficient_dataloader

Data Summary
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   OptimizedPreloadedDataset
   ScenarioSplitDataset
   calculate_complex_statistics
   normalize_complex_matrix
   denormalize_complex_matrix
   create_efficient_dataloader 