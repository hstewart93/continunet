
Quickstart
==========================

The package currently supports `.FITS` type images. To perform source finding, you can import the `finder` module:

.. code-block:: python

   from continunet.finder import Finder

Load your image file:

.. code-block:: python

   finder = Finder("<filepath>")

To produce a source catalogue and populate the `Finder` instance:

.. code-block:: python

   sources = finder.find()

To calculate the model map and residuals image as part of source finding, use `Finder.find()` with `generate_maps=True`:

.. code-block:: python

   sources = finder.find(generate_maps=True)
   model_map = finder.model_map
   residuals = finder.residuals

Alternatively, manually calculate model map and residual images using:

.. code-block:: python

   model_map = finder.get_model_map()
   residuals = finder.get_residuals()

Useful available attributes of the `Finder` object:

.. code-block:: python

   finder.sources
   finder.reconstructed_image
   finder.segmentation_map
   finder.model_map
   finder.residuals
   finder.raw_sources

Export the source catalogue using `finder.export_sources` as `.csv` by default, or `.FITS` by setting `export_fits=True`:

.. code-block:: python

   finder.export_sources("<filepath>", export_fits=<Boolean>)

Source parameters extracted are:

+----------------------------+--------------------------------------------------------------------------------------------------------+
| **Parameter**              | **Description**                                                                                        |
+============================+========================================================================================================+
| x_location_original        | x coordinate of the source from the cutout used for inference                                          |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| y_location_original        | y coordinate of the source from the cutout used for inference                                          |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| orientation                | orientation of source ellipse in radians                                                               |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| major_axis                 | major axis of source ellipse                                                                           |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| minor_axis                 | minor axis of source ellipse                                                                           |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| flux_density_uncorrected   | total intensity of the segmented source region before beam corrections applied                         |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| label                      | class label in predicted segmentation map                                                              |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| x_location                 | x coordinate of the source in the original input image dimensions                                      |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| y_location                 | y coordinate of the source in the original input image dimensions                                      |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| right_ascension            | RA coordinate of the source in the original input image dimensions                                     |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| declination                | Dec coordinate of the source in the original input image dimensions                                    |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| area                       | area of source ellipse                                                                                 |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| position_angle             | position angle of source ellipse in degrees                                                            |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| correction_factor          | correction factor applied to flux density measurement to account for undersampling of synthesised beam |
+----------------------------+--------------------------------------------------------------------------------------------------------+
| flux_density               | corrected flux density                                                                                 |
+----------------------------+--------------------------------------------------------------------------------------------------------+
