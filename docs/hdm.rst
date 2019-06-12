========================
Hausdorff Distance Masks
========================

Introduction
------------

How does it work?
-----------------

Example
-------
.. code-block:: python

    from interpret_segmentation import hdm

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # a PyTorch model
    model = ...

    # a PyTorch dataset
    dataset = ...

    # ground truth segment (PyTorch 2D tensor)
    segment = ...

    # input image  (PyTorch 2D tensor)
    image = ...

    # initialize the explainer with image width and height
    explainer = hdm.HausdorffDistanceMasks(240, 240)

    # generate masks
    explainer.generate_masks(circle_size=25, offset=5)

    # apply masks and calculate distances
    result = explainer.explain(model, image, segment, device)

    # generate
    raw = result.circle_map(hdm.RAW, color_map='Blues')
    better = result.circle_map(hdm.BETTER_ONLY, color_map='Greens')
    worse = result.circle_map(hdm.WORSE_ONLY, color_map='Reds')

    # show with matplotlib...
    plt.imshow(raw)
    plt.show()

    # ...or save to disk
    raw.save('raw.png')

Class documentation
-------------------
.. autoclass:: interpret_segmentation.hdm.HausdorffDistanceMasks
   :members:

.. autoclass:: interpret_segmentation.hdm.HDMResult
   :members:
