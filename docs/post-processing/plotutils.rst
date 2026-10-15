.. _dragons-plotutils:

plotutils
=========

.. py:module:: dragons.plotutils

.. py:function:: density_contour(xdata, ydata, bins, ax, label=True, smooth=0.0, clabel_kwargs={}, **contour_kwargs)

   Create a density contour plot.

   Code modified from:
   https://gist.github.com/adrn/3993992#file-density_contour-py

   :param xdata:
   :type xdata: ndarray
   :param ydata:
   :type ydata: ndarray
   :param bins: Number of bins [nbins_x, nbins_y]. If int then
                nbins_x=nbins_y=nbins.
   :type bins: int or list
   :param ax: Axis to draw contours on
   :type ax: matplotlib.axes.AxesSubplot
   :param label: Draw labels on the contours? (default: True)
   :type label: bool
   :param smooth: Smooth the contours by a gaussian filter with given standard deviation (default: 0.0)
   :type smooth: float
   :param clabel_kwargs: kwargs to be passed to pyplot.clabel() (default: {})
   :type clabel_kwargs: dict
   :param \*\*contour_kwargs: kwargs to be passed to pyplot.contour()
   :type \*\*contour_kwargs: dict

   :returns: **contour**
   :rtype: matplotlib.contour.QuadContourSet

   `Source <https://github.com/meraxes-devs/dragons/blob/b13161739bd20a47c87952cdc89f74af0f7a7e26/dragons/plotutils.py#L10-L83>`__

