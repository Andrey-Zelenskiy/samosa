import numpy as np

def get_rotation_matrix(axis, angle):
    # Normalize the axis of rotation 
    n = axis/np.linalg.norm(axis)
    
    # Calculate cos and sin of the angle
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Populate rotation matrix
    rotation_matrix = np.zeros((3,3))
    
    rotation_matrix[0,0] = cos_a + n[0]**2*( 1 - cos_a )
    rotation_matrix[0,1] = n[0]*n[1]*( 1 - cos_a ) - n[2]*sin_a
    rotation_matrix[0,2] = n[0]*n[2]*( 1 - cos_a ) + n[1]*sin_a
    
    rotation_matrix[1,0] = n[0]*n[1]*( 1 - cos_a ) + n[2]*sin_a
    rotation_matrix[1,1] = cos_a + n[1]**2*(1-cos_a)
    rotation_matrix[1,2] = n[1]*n[2]*( 1 - cos_a ) - n[0]*sin_a

    rotation_matrix[2,0] = n[2]*n[0]*( 1 - cos_a ) - n[1]*sin_a
    rotation_matrix[2,1] = n[2]*n[1]*( 1 - cos_a ) + n[0]*sin_a
    rotation_matrix[2,2] = cos_a + n[2]**2*(1-cos_a)

    return rotation_matrix


def get_orbit(point_0,generators):
    
    ind = 0

    point_index = {}
    point_index[tuple(point_0)] = ind

    operator_map = {}
    operator_map[ind] = np.eye(3)

    orbit = [tuple(point_0)]

    stabilizers = [np.eye(3)]

    for point in orbit:
        for g in generators:
            axis, angle = g
            g_mat = get_rotation_matrix(axis,angle)

            image = tuple(np.round(g_mat.dot(point),4))

            if image not in orbit:
                ind += 1

                orbit += [image]
                point_index[image] = ind
                operator_map[ind] = g_mat.dot(operator_map[point_index[point]])

            elif point_index[image] == 0:
    
                print(image)
                print(np.round(g_mat.dot(operator_map[point_index[point]]),4))
                stabilizers += [g_mat.dot(operator_map[point_index[point]])]


    return orbit, point_index, operator_map, stabilizers

### Point group generators

C2 = [ [ [ 0, 0, 1 ], np.pi ] ]

C4 = [ [ [ 0, 0, 1 ], np.pi/2 ] ]

C6 = [ [ [ 0, 0, 1 ], 2*np.pi/3 ] ]

O  = [ [ [ 0, 0, 1 ],   np.pi/2 ],
       [ [ 1, 1, 1 ], 2*np.pi/3 ] ] 

### Lattice characteristics

chain_l = {"dimension" : 1,
           "basis" : 1
           }

square_l = {"dimension" : 2, 
            "basis" : np.array([ [ 1, 0 ],
                                 [ 0, 1 ] ])
            }

hexagonal_l = {"dimension" : 2, 
               "basis" : np.array([ [  1.0,          0.0 ],
                                    [ -0.5, np.sqrt(3)/2 ] ])
               }

simple_cubic_l = {"dimension" : 3, 
                  "basis" : np.array([ [ 1, 0, 0 ],
                                       [ 0, 1, 0 ],
                                       [ 0, 0, 1 ] ])
                  }

base_centered_cubic_l = {"dimension" : 3, 
                         "basis" : np.array([ [  0.5,  0.5, -0.5 ],
                                              [ -0.5,  0.5,  0.5 ],
                                              [  0.5, -0.5,  0.5 ] ])
                         }

face_centered_cubic_l = {"dimension" : 3,
                         "basis" : np.array([ [ 0.0, 0.5, 0.5 ],
                                              [ 0.5, 0.0, 0.5 ],
                                              [ 0.5, 0.5, 0.0 ] ])
                         }

