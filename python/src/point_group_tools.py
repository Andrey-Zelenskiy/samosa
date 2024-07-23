import numpy as np

def rotation_matrix(axis, angle):
    """
    Returns a 3D matrix describing a rotation around a specified axis by 
    specified angle

    Arguments:
    axis  - array[3], rotation axis
    angle - float,rotation angle

    Returns:
    rotation - array[3][3], rotation matrix
    """

    # Normalize rotation axis
    n = axis/np.linalg.norm(axis)
    
    # Define trig functions
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    rotation = np.zeros((3,3))

    rotation[0,0] = cos_a + n[0]**2*(1-cos_a)
    rotation[0,1] = n[0]*n[1]*( 1 - cos_a ) - u[2]*sin_a 
    rotation[0,2] = n[0]*n[2]*( 1 - cos_a ) + u[1]*sin_a

    rotation[1,0] = n[1]*n[0]*( 1 - cos_a ) + u[2]*sin_a
    rotation[1,1] = cos_a + n[1]**2*(1-cos_a) 
    rotation[1,2] = n[1]*n[2]*( 1 - cos_a ) - u[0]*sin_a
    
    rotation[2,0] = n[2]*n[0]*( 1 - cos_a ) - u[1]*sin_a
    rotation[2,1] = n[2]*n[1]*( 1 - cos_a ) + u[0]*sin_a 
    rotation[2,2] = cos_a + n[2]**2*(1-cos_a)

    return rotation


