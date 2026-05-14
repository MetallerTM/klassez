Make figures with *KLASSEZ*
===========================

The module :mod:`klassez.figures` contains a lot of functions that can be used to generate extremely high quality figures, suitable for publications, that can also be made modular.
They work together with a few functions of :mod:`klassez.misc`, responsible for changing fontsizes, appearance of the axes, and so on.
This section of the guide exists to instruct the user on how to create such figures, in order to adapt them to their needs.

The "engine" we will use for computing the figure is based on the :mod:`matplotlib.pyplot`.
We will explain the basics of computing a figure from scratch by using ``matplotlib``'s default functions, hence they might appear redundant to an expert user.
However, we believe it will be useful in particular for non-developers, i.e. people that *use*, not *write*, the softwares.

The default figure format employed throughout the whole *KLASSEZ* package is the Scalable Vector Graphics (`.svg`).
This choice was made in order to preserve the scalability of the figures without any quality loss, and at the same time keeping the amount of disk space required to the lowest possible amount.
It is of course possible to use bitmap formats: all the figures have the parameter ``ext``, that can be set to whatever figure format the user wants to.

To inspect the pictures, the user must have installed on their computer an image-viewer software.
We recommend the use of ``eog`` ("Eye of Gnome") for linux users, which is lightweight and portable, and has full support for both bitmap and vectorial formats.
On Windows and Mac, where ``eog`` is natively not supported, we recommend to use ``picview`` (`website <https://picview.org/>`_), that is similar to ``eog`` in terms of functionalities.
For further editing of the images (which should not be required in principle, but never say never) we recommend the use of ``inkscape`` (`website <https://inkscape.org/about/>`_) for vectorials and of GIMP (GNU Image Manipulation Program, `website <https://www.gimp.org/>`_) for bitmap.

As a final remark, please keep in mind that the documentation of :mod:`matplotlib` is one of the most extensive and complete of the entire python environment, and it is also full of examples. Make use of it!



The basics: figure panels and subplots
**************************************

If a script has to generate a figure at some point, it must of course import the correct module:
::

    import matplotlib.pyplot as plt

We must always start by instantiating a `figure panel`, i.e. a white canvas that can later host several containers.
It is common practice to keep a hard-reference to it (``fig =``) in order to be able to edit it later.
::

    fig = plt.figure(figure_title)

``figure_title`` is the string that will appear in the top header of the figure window. If nothing is specified, "Figure 1" will appear.
You can set the dimension of the figure with the command
::

    fig.set_size_inches(width, height)

We recommend to use ``15, 8`` for visualization plots and ``8, 6`` for saved figures.
Remember that the fontsizes and thickness of the lines must be tuned accordingly to the dimension of the figure!

The directive 

::

    plt.show()

must be used when we want the figure to be rendered and appear on screen.
The following example code:
::

    fig = plt.figure('TEST')
    fig.set_size_inches(15, 8)

    plt.show()

will generate an empty white rectangle, :math:`15 \times 8` inches wide, with nothing on it.
Very little useful!

In order to make modular figures, and control with very high precision their appearance, we make use of `subplots`, which are containers for `artists` (dots, lines, ticks, text, whatever comes to your mind that goes on a figure is an artist).
There are many ways to add subplots to your figure. We will show you two: ``add_subplot()`` and ``add_axes``.

The general syntax for ``add_subplot`` is 
::

    ax = fig.add_subplot(m, n, k)

which essentially means "divide my figure in a grid that has ``m`` columns and ``n`` rows, number them from 1 to :math:`mn`, take the number ``k``, and reference it as ``ax``".
For example, ``myax = fig.add_subplot(3, 2, 4)`` will create a grid with three columns and two rows, and ``myax`` will be the subplot on the right of the middle row.
Experiment a bit, and read the `documentation of matplotlib <https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.add_subplot.html>`_ to explore the full power of this statement!

The subplots positions with respect to ``fig`` and to each other can be tuned with the directive
::

    plt.subplots_adjust(left, right, top, bottom, wspace, hspace)

The values associated to these parameter can be taken from the "Configure subplots" button in the interactive figure viewer (which is, with the ``qtagg`` framework, the button next to the lens).

``add_axes`` works in a different manner, and for reasons you will clearly understand in a second it is very suitable for insets.
The statement
::

    ax = fig.add_axes([x, y, width, height], transform=transform)

will instantiate as ``ax`` a rectangular subplot with the left bottom corner in (``x``, ``y``), ``width`` large and ``height`` tall.
As opposed to ``add_subplot``, a subplot created with ``add_axes`` is **not** affected by ``plt.subplots_adjust`` in any way.

The numbers we have to write in ``x``, ``y``, ``width``, ``height`` depend on the `coordinate system` we choose, which is declared through the ``transform`` parameter.
We need to define two of them. ``transform=fig.transFigure`` will use the reference frame of ``fig``, which has the origin in the bottom left corner of the figure panel (``(0, 0)``) and the top right corner is ``(1, 1)``; this will be the one we will use to add "regular" subplots.
For other purposes, we can use the reference frame of another subplot (e.g. ``ax1``) by stating ``transform=ax1.transAxes``. This will instruct the backend to use the values that ``ax1`` has **on its axes**.

Customizing subplots
--------------------

In :numref:`f-fig_placeholder` we generated a figure with only one subplot, in order to show which are the main attributes of a subplot and illustrate how to change them.

.. figure:: _static/example_figure_placeholder.png
    :name: f-fig_placeholder

    This is a figure with only one subplot. Each subplot is delimited by four `spines` and has a `title`, two `axes labels`, and certain number of `ticks` with associated `tick labels`.
    

Have you ever seen those very fancy figures of spectra, where there is only the chemical shift scale and the spectrum?
These can be done by setting all the spines invisible except for the bottom one.
This being said, **DON'T DO IT**. Unless you have a *very valid reason* to do so.
If such a valid reason exist and you *really* want to turn the spines off, you can access them via the attribute ``spines`` of the subplot itself, which is a dictionary of four elements: ``'left'``, ``'right'``, ``'bottom'``, ``'top'``.
You can deactivate them with their method ``set_visible``, for example:
::

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

Title and axes labels are quite similar in how they can be set:
::

    ax.set_title(new_title)
    ax.set_xlabel(new_X_label)
    ax.set_ylabel(new_Y_label)

Each of these are :class:`matplotlib.text.Text` objects (`reference <https://matplotlib.org/stable/api/text_api.html>`_), and therefore they accept all the listed keyworded argument for customization.

For customizing the ticks and the tick labels, you need the :func:`matplotlib.axes.Axes.tick_params` function, to be used as:
::

    ax.tick_params(**kwargs)

where all the keyworded arguments allowed are listed in the `documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html>`_.
Nevertheless, they are quite intuitive. If for example one decides that wants to deactivate the ticks and the labels on the left spine, one should write
::

    ax.tick_params(axis='y', left=False, labelleft=False)

Another very important thing one should be able to set is the limits for the axes. 
`matplotlib` has an inner method that sets such limits in order to fit all the artists you draw in a given subplot, with a bit of spacing around.
This might not be what you want!
You can set customized limits for the axes by:
::

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

Keep in mind that the limits follow `exactly` this order, which means that if for example ``left`` is greater than ``right`` the x-axis will be reversed. 
This comes very handy for the chemical shift scales, which as you surely know goes from higher values on the left to lower values on the right.


Plot spectra with *KLASSEZ*
***************************

Now that we made a brief introduction on how `matplotlib` works, you can imagine that setting all these things for every plot one creates might become very cumbersome.
`KLASSEZ` has several functions that are able to create "pre-made" figures, e.g. the function :func:`klassez.figures.figure1D`.
Although these are quite customizable, they do not allow to exploit the full power of `matplotlib`.
This is the reason why there exist a set of functions in the :mod:`klassez.figures` that allow to draw artists inside a given, pre-existing subplot, which are the one whose name starts with ``ax``.
Then, you can use a few function of the :mod:`klassez.misc` to customize the appearance in very few lines.

Learn with examples: superimposed plot of two spectra
-----------------------------------------------------

Let us suppose that we have two spectra, that we will call ``s1`` and ``s2``, both :class:`klassez.Spectra.Spectrum_1D` objects with full processing already performed.
The goal is to create a figure of two subpanels: a left one with the whole spectra, and a right one with highlighted the region between 1 and 0 ppm.
The dimension of the subplots should be the left three times bigger than the right.

How do we do it?

Figure panel and subplots
^^^^^^^^^^^^^^^^^^^^^^^^^

Of course, we will start by creating a figure panel.
We will use the ``add_subplot`` method on a :math:`1 \times 4` grid, and the left panel will merge the first three subplots.
::

    fig = plt.figure()
    fig.set_size_inches(15, 8)
    ax_left = fig.add_subplot(1, 4, (1, 3))
    ax_right = fig.add_subplot(1, 4, 4)


Adding artists
^^^^^^^^^^^^^^

Then, we will put our spectra in both subplots using the :func:`klassez.figure.ax1D` function: ``s1`` in blue, ``s2`` in red.
::

    for ax in [ax_left, ax_right]:
        klassez.figures.ax1D(ax, s1.ppm, s1.r, c='tab:blue', lw=0.8, label='$s_1$')
        klassez.figures.ax1D(ax, s2.ppm, s2.r, c='tab:red', lw=0.8, label='$s_2$')

If you read the documentation of :func:`klassez.figure.ax1D`, you will find that we left many arguments, that might be useful, as default.
This is done on purpose, as we are now going to modify them explicitely.

It might also be the case to add a legend. The entries of the legend gets automatically computed on the basis of the ``label`` attribute set in the various artists.
::

    ax_left.legend(loc='upper left')

Setting the axes limits
^^^^^^^^^^^^^^^^^^^^^^^

:func:`klassez.figure.ax1D` sets automatically the limits of the subplot to the edges of the ppm scale, in the correct sorting order.
This is what we want for the left subplot, but not for the right.
Since we are interested in the 1-0 ppm range, we will just
::

    ax_right.set_xlim(1, 0)

The problem is now that the limits of the y-axis will remain the ones of the full spectrum, which is not good for showing details.
We will then use the function :func:`klassez.misc.set_ylim`, which recalculates the limits of the y-axis on the basis of the current limits.
::

    misc.set_ylim(ax_right, [s1.r, s2.r], s1.ppm, lims=(0, 1))


Cosmetic stuff
^^^^^^^^^^^^^^

At this point, we might want to customize the axes. Let's add the labels, but we want the label of the y-axis to not appear on the right subplot.
::

    for k, ax in enumerate([ax_left, ax_right]):
        ax.set_xlabel(r'$\delta\, ^1$H /ppm')
        if k == 0:
            ax.set_ylabel('Intensity /a.u.')
        else:
            ax.set_ylabel(None)

Then, we need better ticks. Here, a very useful function is :func:`klassez.misc.pretty_scale`, which adds both the major (the one with the numbers) and the minor ticks in a very intelligent way. The syntax is:
::

    for ax in [ax_left, ax_right]:
        # x-axis
        misc.pretty_scale(ax, ax.get_xlim(), 'x', n_major_ticks=10, minor_each=5)
        # y-axis
        misc.pretty_scale(ax, ax.get_ylim(), 'y', n_major_ticks=10, minor_each=5)

``n_major_ticks`` is the `suggested` number of ticks: the actual number will be adapted according to what numbers fit best in the figure. 
Increasing or decreasing this number will affect the appeal of the scale, making it more or less crowded. 
You need to choose the best balance between the accuracy you want and the confusion you create with the numbers.
``minor_each`` will tell the system how many minor ticks you want to be in between two subsequent major ticks. 
The best options are 4 and 5, depending on how the scale is drawn relatively to ``n_major_ticks``.
In this example, we are in a situation in which the left plot is much wider than the right one. 
This means that we will need less ticks on the right subplot with respect to the left one. We will modify the general code above as follows:
::

    for k, ax in enumerate([ax_left, ax_right]): 
        if k == 0: 
            misc.pretty_scale(ax, ax.get_xlim(), 'x', n_major_ticks=16, minor_each=5) 
        else: 
            misc.pretty_scale(ax, ax.get_xlim(), 'x', n_major_ticks=5, minor_each=4) 
        misc.pretty_scale(ax, ax.get_ylim(), 'y', n_major_ticks=10, minor_each=5) 


You can also want to set the y-axis scale in exponential format, in order to not deal with huge numbers.
To do so, use :func:`klassez.misc.mathformat` as
::

    for ax in [ax_left, ax_right]:
        misc.mathformat(ax, limits=(-2, 2))

``limits`` is a tuple of two "exponents" for the base 10: if a value on the scale exceeds :math:`10^p` ``for p in limits``, then the scale becomes exponential, otherwise it stays normal.

Final thing, we must make the fontsizes bigger.
With a figure so big, the default fontsize of 10 will not be enough to read!
The function :func:`klassez.misc.set_fontsizes` will change all the fontsizes in the figure employing an importance gradient: the title will be ``fontsize``, the axes labels will be ``fontsize - 2``, the tick labels will be ``fontsize - 3``, and the legend entries will be ``fontsize - 4``.
::

    for ax in [ax_left, ax_right]:
        misc.set_fontsizes(ax, 20)


If we don't want to play a lot around with ``plt.subplots_adjust``, we can call
::

    fig.tight_layout()

to get rid of the excess empty spaces.

Saving the figure
^^^^^^^^^^^^^^^^^

At this point the only thing we are left with is to save the figure (or show it with ``plt.show()``).
To save a figure, the most extensive block of code you could write is the following:
::
    
    filename = 'myfigure'
    ext = 'svg'
    dpi = 300

    plt.savefig(Path(filename).with_suffix(f'.{ext}'), dpi=dpi)
    plt.close()

With the proper extension, the format is recognized automatically. Check :func:`matplotlib.pyplot.savefig` for extra details and for the supported formats for the figures.
The ``dpi`` parameter is useful to define the resolution of the figure, so that it can fit an article with dignity.

When you are done rendering the figure and you saved it, remember always to ``plt.close()`` it, otherwise it will linger in the memory occupying useful space you might need for extra tasks.
Consider that ``plt.show()`` has the ``plt.close()`` instruction inside when you close the panel!
This means that calling for ``plt.savefig`` *after* ``plt.show()`` will save **nothing**!.


Example script: superimposed plot of two spectra
------------------------------------------------

The spectra are simulated with ``sim1.acqus`` and ``sim2.acqus``, that you find here below together with the script to process them.

::

    # Figure panel and subplots
    fig = plt.figure()
    fig.set_size_inches(15, 8)
    ax_left = fig.add_subplot(1, 4, (1, 3))
    ax_right = fig.add_subplot(1, 4, 4)

    # Draw both spectra in both subplots
    for ax in [ax_left, ax_right]:
        figures.ax1D(ax, s1.ppm, s1.r, c='tab:blue', lw=0.8, label='$s_1$')
        figures.ax1D(ax, s2.ppm, s2.r, c='tab:red', lw=0.8, label='$s_2$')

    # Add the legend
    ax_left.legend(loc='upper left')

    # Set limits for the axes
    ax_right.set_xlim(1, 0)
    misc.set_ylim(ax_right, [s1.r, s2.r], s1.ppm, lims=(0, 1))

    # Cosmetic stuff
    for k, ax in enumerate([ax_left, ax_right]):
        # Label of x-axis
        ax.set_xlabel(r'$\delta\, ^1$H /ppm')

        if k == 0:  # => only left
            # Label of y-axis
            ax.set_ylabel('Intensity /a.u.')
            # x-scale with a lot of ticks
            misc.pretty_scale(ax, ax.get_xlim(), 'x', n_major_ticks=16, minor_each=5)
        else:       # => only right
            # Remove the label of the y-axis
            ax.set_ylabel(None)
            # x-scale with less ticks
            misc.pretty_scale(ax, ax.get_xlim(), 'x', n_major_ticks=5, minor_each=4)
        # y-scale is (from the cosmetic side) the same for both subplots
        misc.pretty_scale(ax, ax.get_ylim(), 'y', n_major_ticks=10, minor_each=5)
        # Exponential format
        misc.mathformat(ax, limits=(-2, 2))
        # Increase fontsizes
        misc.set_fontsizes(ax, 20)

    # Reduce whitespaces
    fig.tight_layout()

    # Save the figure
    filename = 'myfigure'
    ext = 'svg'
    dpi = 300

    plt.savefig(Path(filename).with_suffix(f'.{ext}'), dpi=dpi)
    plt.close()


Input files
^^^^^^^^^^^
::

    # sim1.acqus

    B0	16.4
    nuc	1H
    o1p	5
    SWp	30
    TD	2**15

    shifts	10, 8, 6, 0.5
    fwhm	20, 20, 20, 20
    amplitudes	1, 2, 2, 0.5,
    b	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    mult	s, s, t, t,
    Jconst	0, 0, [50.5, 1.5], [50.5, 1.5],

::

    # sim2.acqus

    B0	16.4
    nuc	1H
    o1p	5
    SWp	30
    TD	2**15

    shifts	10, 8, 6, 0.5
    fwhm	20, 20, 20, 20
    amplitudes	0.5, 2.5, 1.5, 0.33,
    b	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    mult	s, s, t, t,  
    Jconst	0, 0, [50.5, 1.5], [50.5, 1.5], 

Processing script
^^^^^^^^^^^^^^^^^
::

    s1 = Spectrum_1D('sim1.acqus', isexp=False)
    s2 = Spectrum_1D('sim2.acqus', isexp=False)
    for s in [s1, s2]:
        s.procs['zf'] = 2*s.fid.shape[-1]
        s.process()






    
