Domain adaptation based on optimal transport
--------------------------------------------
Python package to simplify OT-based domain adaptation. This package uses [POT](https://pythonot.github.io/auto_examples/domain-adaptation/plot_otda_color_images.html) as optimal transport backend.

Install with pip
----------------

```bash
$ python3 -m pip install otda --user
```

Install from source
-------------------

```bash
$ python3 setup.py install --user
```

Exemplary code snippet
----------------------

```python                                                                                                      
adapted_im = otda.colour_transfer(source_im, target_im, method='linear', nsamples=1000)
```

Run domain adaptation on a single image
---------------------------------------

```bash
$ python3 -m otda.run --source source.jpg --target target.jpg --output output.jpg --method emd
```
Available methods: linear, linear_fourier, gaussian, sinkhorn, emd.


License
-------

This repository is shared under an [MIT license](https://github.com/luiscarlosgph/ot-domain-adaptation/blob/main/LICENSE).


Author
------

Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com), 2020-2022.
