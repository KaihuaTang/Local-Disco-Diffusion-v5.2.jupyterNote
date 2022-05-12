import pandas as pd
import numpy as np



def parse_key_frames(string, prompt_parser=None):
    """Given a string representing frame numbers paired with parameter values at that frame,
    return a dictionary with the frame numbers as keys and the parameter values as the values.

    Parameters
    ----------
    string: string
        Frame numbers paired with parameter values at that frame number, in the format
        'framenumber1: (parametervalues1), framenumber2: (parametervalues2), ...'
    prompt_parser: function or None, optional
        If provided, prompt_parser will be applied to each string of parameter values.
    
    Returns
    -------
    dict
        Frame numbers as keys, parameter values at that frame number as values

    Raises
    ------
    RuntimeError
        If the input string does not match the expected format.
    
    Examples
    --------
    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)")
    {10: 'Apple: 1| Orange: 0', 20: 'Apple: 0| Orange: 1| Peach: 1'}

    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)", prompt_parser=lambda x: x.lower()))
    {10: 'apple: 1| orange: 0', 20: 'apple: 0| orange: 1| peach: 1'}
    """
    import re
    pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param

    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames


def get_inbetweens(key_frames, max_frames, interp_spline, integer=False):
    """Given a dict with frame numbers as keys and a parameter value as values,
    return a pandas Series containing the value of the parameter at every frame from 0 to max_frames.
    Any values not provided in the input dict are calculated by linear interpolation between
    the values of the previous and next provided frames. If there is no previous provided frame, then
    the value is equal to the value of the next provided frame, or if there is no next provided frame,
    then the value is equal to the value of the previous provided frame. If no frames are provided,
    all frame values are NaN.

    Parameters
    ----------
    key_frames: dict
        A dict with integer frame numbers as keys and numerical values of a particular parameter as values.
    integer: Bool, optional
        If True, the values of the output series are converted to integers.
        Otherwise, the values are floats.
    
    Returns
    -------
    pd.Series
        A Series with length max_frames representing the parameter values for each frame.
    
    Examples
    --------
    >>> max_frames = 5
    >>> get_inbetweens({1: 5, 3: 6})
    0    5.0
    1    5.0
    2    5.5
    3    6.0
    4    6.0
    dtype: float64

    >>> get_inbetweens({1: 5, 3: 6}, integer=True)
    0    5
    1    5
    2    5
    3    6
    4    6
    dtype: int64
    """
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)
    
    interp_method = interp_spline

    if interp_method == 'Cubic' and len(key_frames.items()) <=3:
      interp_method = 'Quadratic'
    
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
      interp_method = 'Linear'
      
    
    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
    # key_frame_series = key_frame_series.interpolate(method=intrp_method,order=1, limit_direction='both')
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(),limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series



def split_prompts(prompts, max_frames):
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series


def update_parameters(key_frames, max_frames, interp_spline, angle, zoom, translation_x, translation_y, translation_z,
                      rotation_3d_x, rotation_3d_y, rotation_3d_z):
    if key_frames:
        try:
            angle_series = get_inbetweens(parse_key_frames(angle), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `angle` correctly for key frames.\n"
                "Attempting to interpret `angle` as "
                f'"0: ({angle})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            angle = f"0: ({angle})"
            angle_series = get_inbetweens(parse_key_frames(angle), max_frames, interp_spline)

        try:
            zoom_series = get_inbetweens(parse_key_frames(zoom), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `zoom` correctly for key frames.\n"
                "Attempting to interpret `zoom` as "
                f'"0: ({zoom})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            zoom = f"0: ({zoom})"
            zoom_series = get_inbetweens(parse_key_frames(zoom), max_frames, interp_spline)

        try:
            translation_x_series = get_inbetweens(parse_key_frames(translation_x), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `translation_x` correctly for key frames.\n"
                "Attempting to interpret `translation_x` as "
                f'"0: ({translation_x})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            translation_x = f"0: ({translation_x})"
            translation_x_series = get_inbetweens(parse_key_frames(translation_x), max_frames, interp_spline)

        try:
            translation_y_series = get_inbetweens(parse_key_frames(translation_y), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `translation_y` correctly for key frames.\n"
                "Attempting to interpret `translation_y` as "
                f'"0: ({translation_y})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            translation_y = f"0: ({translation_y})"
            translation_y_series = get_inbetweens(parse_key_frames(translation_y), max_frames, interp_spline)

        try:
            translation_z_series = get_inbetweens(parse_key_frames(translation_z), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `translation_z` correctly for key frames.\n"
                "Attempting to interpret `translation_z` as "
                f'"0: ({translation_z})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            translation_z = f"0: ({translation_z})"
            translation_z_series = get_inbetweens(parse_key_frames(translation_z), max_frames, interp_spline)

        try:
            rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `rotation_3d_x` correctly for key frames.\n"
                "Attempting to interpret `rotation_3d_x` as "
                f'"0: ({rotation_3d_x})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            rotation_3d_x = f"0: ({rotation_3d_x})"
            rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x), max_frames, interp_spline)

        try:
            rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `rotation_3d_y` correctly for key frames.\n"
                "Attempting to interpret `rotation_3d_y` as "
                f'"0: ({rotation_3d_y})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            rotation_3d_y = f"0: ({rotation_3d_y})"
            rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y), max_frames, interp_spline)

        try:
            rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `rotation_3d_z` correctly for key frames.\n"
                "Attempting to interpret `rotation_3d_z` as "
                f'"0: ({rotation_3d_z})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            rotation_3d_z = f"0: ({rotation_3d_z})"
            rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z), max_frames, interp_spline)

    else:
        angle = float(angle)
        zoom = float(zoom)
        translation_x = float(translation_x)
        translation_y = float(translation_y)
        translation_z = float(translation_z)
        rotation_3d_x = float(rotation_3d_x)
        rotation_3d_y = float(rotation_3d_y)
        rotation_3d_z = float(rotation_3d_z)
        
    series_params = (angle_series, zoom_series, translation_x_series, translation_y_series, translation_z_series,
                     rotation_3d_x_series, rotation_3d_y_series, rotation_3d_z_series)
    float_params = (angle, zoom, translation_x, translation_y, translation_z,
                    rotation_3d_x, rotation_3d_y, rotation_3d_z)
    return series_params, float_params