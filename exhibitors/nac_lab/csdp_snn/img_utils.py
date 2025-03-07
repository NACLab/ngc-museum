import itertools
from jax import numpy as jnp, random, jit, vmap
import functools
from functools import partial

def solarize(image, threshold):
  """Applies solarization to an image.

  All values above a given threshold will be inverted.

  Args:
    image: an RGB image, given as a [0-1] float tensor.
    threshold: the threshold for inversion.

  Returns:
    The solarized image.
  """
  return jnp.where(image < threshold, image, 1. - image)

def gen_neg_indices(dkey, n):
    indices = []
    start = 0
    end = n #n-1
    for i in range(n):
        if i == 0:
            choices = list(range(i+1, end))
        elif i == n-1:
            choices = list(range(start, i))
        else:
            choices = list(range(start, i)) + list(range(i+1, end))
        idx = random.choice(dkey, jnp.asarray(choices))
        #idx = random.randint(ptr+1,n)
        indices.append(idx)
    indices = jnp.asarray(indices)
    return indices

@jit
def _interpolate(x_orig, x_new, alpha):
    return x_orig * alpha + x_new * (1. - alpha)

def csdp_deform(dkey, x_pos, y_pos, alpha=0.5, use_rot=True): ## neg sample synthesizer for CSDP
    #_x_pos = alter(x_pos) ## arbitrarily transform x_pos
    neg_ind = gen_neg_indices(dkey, x_pos.shape[0])
    x_neg = x_pos[neg_ind,:] # select opposite image patterns
    y_neg = y_pos[neg_ind,:]

    _x_neg = x_neg
    if use_rot == True:
        #_x_neg = jnp.expand_dims(jnp.reshape(x_neg, [x_pos.shape[0], 28, 28]), axis=3)
        rot = random.uniform(dkey, (x_pos.shape[0],), minval=-90, maxval=90)
        _x_neg = vrotate(_x_neg, rot)
    # shift = tf.random.uniform([x_pos.shape[0], 2], -3., 3.)
    # _x_neg = translate(_x_neg, shift)
    #_x_neg = jnp.squeeze(jnp.reshape(_x_neg, [x_pos.shape[0], 28*28, 1]))

    x_neg = _interpolate(x_pos, _x_neg, alpha)
    #x_neg = x_pos * alpha + _x_neg * (1. - alpha)
    return x_neg, y_neg

def rand_rotate(dkey, xB, minTheta=-28.6478898, maxTheta=28.6478898):
    _angles = random.uniform(dkey, (xB.shape[0],), minval=minTheta, maxval=maxTheta)
    xB_R = vmap(rotate, in_axes=(0,0))(xB, _angles)
    return xB_R

def vrotate(xB, angles):
    _angles = angles
    if len(angles) == 1: ## smear scalar into list
        angle = angles[0]
        #_angles = angles * xB.shape[0]
        xB_R = vmap(rotate, in_axes=(0, None))(xB, angle)
    else:
        _angles = jnp.asarray(_angles)
        xB_R = vmap(rotate, in_axes=(0,0))(xB, _angles)
    return xB_R

def rotate(
    image,
    angle, # in degrees
    *,
    order=1,
    mode="nearest", #"constant", #"nearest",
    cval=0.0,
):
  """Rotates an image around its center using interpolation.

  Args:
    image: a JAX array representing an image. Assumes that the image is
      either HWC or CHW.
    angle: the counter-clockwise rotation angle in units of degrees.
    order: the order of the spline interpolation, default is 1. The order has
      to be in the range [0,1]. See `affine_transform` for details.
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Default is 'nearest'. See `affine_transform` for details.
    cval: value to fill past edges of input if mode is 'constant'. Default is
      0.0.

  Returns:
    The rotated image.
  """
  angle_rad = angle * (jnp.pi/180.) #-angle * (jnp.pi/180.)
  return _rotate(image=image, angle=angle_rad, order=order, mode=mode, cval=cval)

@partial(jit, static_argnums=[1, 2, 3, 4])
def _rotate(
    image,
    angle,
    order=1,
    mode="nearest",
    cval=0.0,
):
  """Rotates an image around its center using interpolation.

  Args:
    image: a JAX array representing an image. Assumes that the image is
      either HWC or CHW.
    angle: the counter-clockwise rotation angle in units of radians.
    order: the order of the spline interpolation, default is 1. The order has
      to be in the range [0,1]. See `affine_transform` for details.
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Default is 'nearest'. See `affine_transform` for details.
    cval: value to fill past edges of input if mode is 'constant'. Default is
      0.0.

  Returns:
    The rotated image.
  """
  # Calculate inverse transform matrix assuming clockwise rotation.
  c = jnp.cos(angle)
  s = jnp.sin(angle)
  matrix = jnp.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

  # Use the offset to place the rotation at the image center.
  image_center = (jnp.asarray(image.shape) - 1.) / 2.
  offset = image_center - matrix @ image_center

  return affine_transform(image, matrix, offset=offset, order=order, mode=mode,
                          cval=cval)

def affine_transform(
    image,
    matrix,
    *,
    offset=0., #Union[chex.Array, chex.Numeric]
    order=1,
    mode="nearest",
    cval=0.0,
):
  """Applies an affine transformation given by matrix.

  Given an output image pixel index vector o, the pixel value is determined from
  the input image at position jnp.dot(matrix, o) + offset.

  This does 'pull' (or 'backward') resampling, transforming the output space to
  the input to locate data. Affine transformations are often described in the
  'push' (or 'forward') direction, transforming input to output. If you have a
  matrix for the 'push' transformation, use its inverse (jax.numpy.linalg.inv)
  in this function.

  Args:
    image: a JAX array representing an image. Assumes that the image is
      either HWC or CHW.
    matrix: the inverse coordinate transformation matrix, mapping output
      coordinates to input coordinates. If ndim is the number of dimensions of
      input, the given matrix must have one of the following shapes:

      - (ndim, ndim): the linear transformation matrix for each output
        coordinate.
      - (ndim,): assume that the 2-D transformation matrix is diagonal, with the
        diagonal specified by the given value.
      - (ndim + 1, ndim + 1): assume that the transformation is specified using
        homogeneous coordinates [1]. In this case, any value passed to offset is
        ignored.
      - (ndim, ndim + 1): as above, but the bottom row of a homogeneous
        transformation matrix is always [0, 0, 0, 1], and may be omitted.

    offset: the offset into the array where the transform is applied. If a
      float, offset is the same for each axis. If an array, offset should
      contain one value for each axis.
    order: the order of the spline interpolation, default is 1. The order has
      to be in the range [0-1]. Note that PIX interpolation will only be used
      for order=1, for other values we use `jax.scipy.ndimage.map_coordinates`.
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Default is 'nearest'. Modes 'nearest and 'constant' use
      PIX interpolation, which is very fast on accelerators (especially on
      TPUs). For all other modes, 'wrap', 'mirror' and 'reflect', we rely
      on `jax.scipy.ndimage.map_coordinates`, which however is slow on
      accelerators, so use it with care.
    cval: value to fill past edges of input if mode is 'constant'. Default is
      0.0.

  Returns:
    The input image transformed by the given matrix.

  Example transformations:
    Rotation:

    >>> angle = jnp.pi / 4
    >>> matrix = jnp.array([
    ...    [jnp.cos(rotation), -jnp.sin(rotation), 0],
    ...    [jnp.sin(rotation), jnp.cos(rotation), 0],
    ...    [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)

    Translation can be expressed through either the matrix itself
    or the offset parameter:

    >>> matrix = jnp.array([
    ...   [1, 0, 0, 25],
    ...   [0, 1, 0, 25],
    ...   [0, 0, 1, 0],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
    >>> # Or with offset:
    >>> matrix = jnp.array([
    ...   [1, 0, 0],
    ...   [0, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> offset = jnp.array([25, 25, 0])
    >>> result = dm_pix.affine_transform(
            image=image, matrix=matrix, offset=offset)

    Reflection:

    >>> matrix = jnp.array([
    ...   [-1, 0, 0],
    ...   [0, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)

    Scale:

    >>> matrix = jnp.array([
    ...   [2, 0, 0],
    ...   [0, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)

    Shear:

    >>> matrix = jnp.array([
    ...   [1, 0.5, 0],
    ...   [0.5, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)

    One can also combine different transformations matrices:

    >>> matrix = rotation_matrix.dot(translation_matrix)
  """
  # DO NOT REMOVE - Logging usage.

  # chex.assert_rank(image, 3)
  # chex.assert_rank(matrix, {1, 2})
  # chex.assert_rank(offset, {0, 1})

  if matrix.ndim == 1:
    matrix = jnp.diag(matrix)

  if matrix.shape not in [(3, 3), (4, 4), (3, 4)]:
    error_msg = (
        "Expected matrix shape must be one of (ndim, ndim), (ndim,)"
        "(ndim + 1, ndim + 1) or (ndim, ndim + 1) being ndim the image.ndim. "
        f"The affine matrix provided has shape {matrix.shape}.")
    raise ValueError(error_msg)

  meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in image.shape],
                          indexing="ij")
  indices = jnp.concatenate(
      [jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)

  if matrix.shape == (4, 4) or matrix.shape == (3, 4):
    offset = matrix[:image.ndim, image.ndim]
    matrix = matrix[:image.ndim, :image.ndim]

  coordinates = indices @ matrix.T
  coordinates = jnp.moveaxis(coordinates, source=-1, destination=0)

  # Alter coordinates to account for offset.
  offset = jnp.full((3,), fill_value=offset)
  coordinates += jnp.reshape(a=offset, newshape=(*offset.shape, 1, 1, 1))

  interpolate_function = _get_interpolate_function(
      mode=mode,
      order=order,
      cval=cval,
  )
  return interpolate_function(image, coordinates)

def _get_interpolate_function(
    mode,
    order,
    cval=0.,
):
  """Selects the interpolation function to use based on the given parameters.

  PIX interpolations are preferred given they are faster on accelerators. For
  the cases where such interpolation is not implemented by PIX we really on
  jax.scipy.ndimage.map_coordinates. See specifics below.

  Args:
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Modes 'nearest and 'constant' use PIX interpolation, which
      is very fast on accelerators (especially on TPUs). For all other modes,
      'wrap', 'mirror' and 'reflect', we rely on
      `jax.scipy.ndimage.map_coordinates`, which however is slow on
      accelerators, so use it with care.
    order: the order of the spline interpolation. The order has to be in the
      range [0, 1]. Note that PIX interpolation will only be used for order=1,
      for other values we use `jax.scipy.ndimage.map_coordinates`.
    cval: value to fill past edges of input if mode is 'constant'.

  Returns:
    The selected interpolation function.
  """
  if mode == "nearest" and order == 1:
    interpolate_function = flat_nd_linear_interpolate
  elif mode == "constant" and order == 1:
    interpolate_function = functools.partial(
        flat_nd_linear_interpolate_constant, cval=cval)
  else:
    interpolate_function = functools.partial(
        jax.scipy.ndimage.map_coordinates, mode=mode, order=order, cval=cval)
  return interpolate_function

def flat_nd_linear_interpolate(
    volume,
    coordinates,
    *,
    unflattened_vol_shape=None, #Optional[Sequence[int]]
):
  """Maps the input ND volume to coordinates by linear interpolation.

  Args:
    volume: A volume (flat if `unflattened_vol_shape` is provided) where to
      query coordinates.
    coordinates: An array of shape (N, M_coordinates). Where M_coordinates can
      be M-dimensional. If M_coordinates == 1, then `coordinates.shape` can
      simply be (N,), e.g. if N=3 and M_coordinates=1, this has the form (z, y,
      x).
    unflattened_vol_shape: The shape of the `volume` before flattening. If
      provided, then `volume` must be pre-flattened.

  Returns:
    The resulting mapped coordinates. The shape of the output is `M_coordinates`
    (derived from `coordinates` by dropping the first axis).
  """
  if unflattened_vol_shape is None:
    unflattened_vol_shape = volume.shape
    volume = volume.flatten()

  indices, weights = _make_linear_interpolation_indices_flat_nd(
      coordinates, shape=unflattened_vol_shape)
  return _linear_interpolate_using_indices_nd(
      jnp.asarray(volume), indices, weights)

def _make_linear_interpolation_indices_flat_nd(
    coordinates,
    shape,
):
  """Creates flat linear interpolation indices and weights for ND coordinates.

  Args:
    coordinates: An array of shape (N, M_coordinates).
    shape: The shape of the ND volume, e.g. if N=3 shape=(dim_z, dim_y, dim_x).

  Returns:
    The indices into the flattened input and their weights.
  """
  coordinates = jnp.asarray(coordinates)
  shape = jnp.asarray(shape)

  if shape.shape[0] != coordinates.shape[0]:
    raise ValueError(
        (f"{coordinates.shape[0]}-dimensional coordinates provided for "
         f"{shape.shape[0]}-dimensional input"))

  lower_nd, upper_nd, weights_nd = _make_linear_interpolation_indices_nd(
      coordinates, shape)

  # Here we want to translate e.g. a 3D-disposed indices to linear ones, since
  # we have to index on the flattened source, so:
  # flat_idx = shape[1] * shape[2] * z_idx + shape[2] * y_idx + x_idx

  # The `strides` of a `shape`-sized array tell us how many elements we have to
  # skip to move to the next position along a certain axis in that array.
  # For example, for a shape=(5,4,2) we have to skip 1 value to move to the next
  # column (3rd axis), 2 values to move to get to the same position in the next
  # row (2nd axis) and 4*2=8 values to move to get to the same position on the
  # 1st axis.
  strides = jnp.concatenate([jnp.cumprod(shape[:0:-1])[::-1], jnp.array([1])])

  # Array of 2^n rows where the ith row is the binary representation of i.
  binary_array = jnp.array(
      list(itertools.product([0, 1], repeat=shape.shape[0])))

  # Expand dimensions to allow broadcasting `strides` and `binary_array` to
  # every coordinate.
  # Expansion size is equal to the number of dimensions of `coordinates` - 1.
  strides = strides.reshape(strides.shape + (1,) * (coordinates.ndim - 1))
  binary_array = binary_array.reshape(binary_array.shape + (1,) *
                                      (coordinates.ndim - 1))

  lower_1d = lower_nd * strides
  upper_1d = upper_nd * strides

  point_weights = []
  point_indices = []

  for r in binary_array:
    # `point_indices` is defined as:
    # `jnp.matmul(binary_array, upper) + jnp.matmul(1-binary_array, lower)`
    # however, to date, that implementation turns out to be slower than the
    # equivalent following one.
    point_indices.append(jnp.sum(upper_1d * r + lower_1d * (1 - r), axis=0))
    point_weights.append(
        jnp.prod(r * weights_nd + (1 - r) * (1 - weights_nd), axis=0))
  return jnp.stack(point_indices, axis=0), jnp.stack(point_weights, axis=0)

def _linear_interpolate_using_indices_nd(
    volume,
    indices,
    weights,
):
  """Interpolates linearly on `volume` using `indices` and `weights`."""
  target = jnp.sum(weights * volume[indices], axis=0)
  if jnp.issubdtype(volume.dtype, jnp.integer):
    target = _round_half_away_from_zero(target)
  return target.astype(volume.dtype)

def _round_half_away_from_zero(a):
  return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)

def flat_nd_linear_interpolate_constant(
    volume,
    coordinates,
    *,
    cval=0.,
    unflattened_vol_shape=None,
):
  """Maps volume by interpolation and returns a constant outside boundaries.

  Maps the input ND volume to coordinates by linear interpolation, but returns
  a constant value if the coordinates fall outside the volume boundary.

  Args:
    volume: A volume (flat if `unflattened_vol_shape` is provided) where to
      query coordinates.
    coordinates: An array of shape (N, M_coordinates). Where M_coordinates can
      be M-dimensional. If M_coordinates == 1, then `coordinates.shape` can
      simply be (N,), e.g. if N=3 and M_coordinates=1, this has the form (z, y,
      x).
    cval: A constant value to map to for coordinates that fall outside
      the volume boundaries.
    unflattened_vol_shape: The shape of the `volume` before flattening. If
      provided, then `volume` must be pre-flattened.

  Returns:
    The resulting mapped coordinates. The shape of the output is `M_coordinates`
    (derived from `coordinates` by dropping the first axis).
  """
  # DO NOT REMOVE - Logging usage.

  volume_shape = volume.shape
  if unflattened_vol_shape is not None:
    volume_shape = unflattened_vol_shape

  # Initialize considering all coordinates within the volume and loop through
  # boundaries.
  is_in_bounds = jnp.full(coordinates.shape[1:], True)
  for dim, dim_size in enumerate(volume_shape):
    is_in_bounds = jnp.logical_and(is_in_bounds, coordinates[dim] >= 0)
    is_in_bounds = jnp.logical_and(is_in_bounds,
                                   coordinates[dim] <= dim_size - 1)

  return flat_nd_linear_interpolate(
      volume,
      coordinates,
      unflattened_vol_shape=unflattened_vol_shape
  ) * is_in_bounds + (1. - is_in_bounds) * cval

def _make_linear_interpolation_indices_nd(
    coordinates,
    shape,
):
  """Creates linear interpolation indices and weights for ND coordinates.

  Args:
    coordinates: An array of shape (N, M_coordinates).
    shape: The shape of the ND volume, e.g. if N=3 shape=(dim_z, dim_y, dim_x).

  Returns:
    The lower and upper indices of `coordinates` and their weights.
  """
  lower = jnp.floor(coordinates).astype(jnp.int32)
  upper = jnp.ceil(coordinates).astype(jnp.int32)
  weights = coordinates - lower

  # Expand dimensions for `shape` to allow broadcasting it to every coordinate.
  # Expansion size is equal to the number of dimensions of `coordinates` - 1.
  shape = shape.reshape(shape.shape + (1,) * (coordinates.ndim - 1))

  lower = jnp.clip(lower, 0, shape - 1)
  upper = jnp.clip(upper, 0, shape - 1)

  return lower, upper, weights
