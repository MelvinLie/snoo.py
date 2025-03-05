import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import numpy.matlib as matlib

class QuadGeometry():
    '''This is a class to model a quadrangle geometry.
    '''

    def __init__(self, p1, p2, p3, p4):
        '''The default constructor.

        :param points:
            The corner points of the quadrangle.
        '''
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def eval(self, u, v):
        '''Evaluate this quad geometry.
        
        :param u:
            The local u coordinate in [-1, 1]
            
        :param v:
            The local v coordinate in [-1, 1]
            
        :return:
            The position.
        '''

        return self.p1*0.25*(u + 1)*(v + 1) \
                    + self.p2*0.25*(1 - u)*(v + 1)\
                          + self.p3*0.25*(1 - u)*(1 - v)\
                              + self.p4*0.25*(u + 1)*(1 - v)

    def eval_normal(self, u, v):
        '''Evaluate the normal vector of this geometry
        
        :param u:
            The local u coordinate in [-1, 1]
            
        :param v:
            The local v coordinate in [-1, 1]
            
        :return:
            The position.
        '''
        
        du = self.p1*0.25*(v + 1) \
                    - self.p2*0.25*(v + 1)\
                          - self.p3*0.25*(1 - v)\
                              + self.p4*0.25*(1 - v)
        
        dv = self.p1*0.25*(u + 1) \
                    + self.p2*0.25*(1 - u)\
                          - self.p3*0.25*(1 - u)\
                              - self.p4*0.25*(u + 1)
        
        n = np.zeros((3, ))
        n[0] = du[1]*dv[2] - du[2]*dv[1]
        n[1] = du[2]*dv[0] - du[0]*dv[2]
        n[2] = du[0]*dv[1] - du[1]*dv[0]

        n_magn = np.linalg.norm(n)
        n[0] /= n_magn
        n[1] /= n_magn
        n[2] /= n_magn
        
        return n
    
    def compute_area(self):
        '''Compute the area of this quad geometry.
        
        The area as a floart.
        '''

        p1 = self.eval(0.0, 0.0)
        p2 = self.eval(1.0, 0.0)
        p3 = self.eval(0.0, 1.0)

        d1 = p2 - p1
        d2 = p3 - p1

        return 0.5*np.linalg.norm(np.cross(d1, d2))

    def plot(self, axes):
        '''Plot the geometry in matplotlib.
        
        :param axes:

            A matplotlib axes object.

        :return:
            None.

        '''
        axes.plot([self.p1[0], self.p2[0], self.p3[0], self.p4[0], self.p1[0]],
                  [self.p1[1], self.p2[1], self.p3[1], self.p4[1], self.p1[1]],
                  [self.p1[2], self.p2[2], self.p3[2], self.p4[2], self.p1[2]], color='k')
        
        ctr = self.eval(0., 0.)
        n = self.eval_normal(0., 0.)

        axes.plot([ctr[0], ctr[0] + 0.1*n[0]],
                  [ctr[1], ctr[1] + 0.1*n[1]],
                  [ctr[2], ctr[2] + 0.1*n[2]], color='red')

class BrickGeometry():
    '''This is a class to model a brick geometry.
    '''

    def __init__(self, points):
        '''The default constructor.

        :param points:
            The corner points of the brick.
        '''

        # it consists of six sides
        self.sides = []

        self.sides.append(QuadGeometry(points[3, :], points[2, :], points[1, :], points[0, :]))
        self.sides.append(QuadGeometry(points[4, :], points[5, :], points[6, :], points[7, :]))

        self.sides.append(QuadGeometry(points[5, :], points[4, :], points[0, :], points[1, :]))
        self.sides.append(QuadGeometry(points[6, :], points[5, :], points[1, :], points[2, :]))

        self.sides.append(QuadGeometry(points[2, :], points[3, :], points[7, :], points[6, :]))
        self.sides.append(QuadGeometry(points[3, :], points[0, :], points[4, :], points[7, :]))

        self.field_vec = np.zeros((3, ))

    def set_field_vector(self, vec):
        '''Setup the field vector.

        :param vec:
            The vector.

        :return:
            None.
        '''
        self.field_vec = vec

        return

    def plot(self, axes):
        '''Plot the geometry in matplotlib.
        
        :param axes:

            A matplotlib axes object.

        :return:
            None.

        '''
        for s in self.sides:
            s.plot(axes)

        return

    def is_inside(self, point):
        '''check if a point is inside (true) or
        outside (false)
        
        :param point:
            The field point.

        :return:
            A flag specifying if it is in or outside.
        '''

        for s in self.sides:

            # the center point
            r = s.eval(0., 0.)

            # the normal vector
            n = s.eval_normal(0., 0.)

            # the difference vector
            diff = point - r

            # the projection onto n
            proj = np.sum(diff*n)
            
            if proj > 0.0:
                return False
            
        return True


    def are_inside(self, points):
        '''check if points in a cloud are inside (true) or
        outside (false)
        
        :param points:
            The field points.

        :return:
            A numpy array of bool type, specifying if the 
            points are inside or outside.
        '''

        mask = np.ones((points.shape[0]), dtype=np.bool_)

        for s in self.sides:

            # the center point
            r = s.eval(0., 0.)

            # the normal vector
            n = s.eval_normal(0., 0.)

            # the difference vector
            diff = points.copy()

            diff[:, 0] -= r[0]
            diff[:, 1] -= r[1]
            diff[:, 2] -= r[2]

            # the projections onto n
            dn = diff.copy()
            dn[:, 0] *= n[0]
            dn[:, 1] *= n[1]
            dn[:, 2] *= n[2]

            proj = np.sum(dn, axis=1)
            
            mask *= proj < 0.0
            
        return mask

    def evaluate_fields(self, points):
        '''Evaluate the fields in points of a cloud

        :param points:
            The position where to evaluate.

        :return:
            The field vector.
        '''

        ret_vec = np.zeros((points.shape[0], 3))
        ret_vec[:, 0] = self.field_vec[0]
        ret_vec[:, 1] = self.field_vec[1]
        ret_vec[:, 2] = self.field_vec[2]

        mask = self.are_inside(points)

        ret_vec[mask == False] *= 0.0

        return ret_vec
    


    def evaluate_field(self, pos):
        '''Evaluate the field in the geometry.

        :param pos:
            The position where to evaluate.

        :return:
            The field vector.
        '''


        if self.is_inside(pos):
            return self.field_vec
        else:
            return np.zeros((3, ))
    

class BrickField():
    '''This is a class to model brick field magnets.
    '''

    def __init__(self, X_mgap_1, X_core_1, X_void_1, X_yoke_1,
                        X_mgap_2, X_core_2, X_void_2, X_yoke_2,
                        Y_void_1, Y_yoke_1,
                        Y_void_2, Y_yoke_2,
                        Z_len, Z_pos=0.0,
                        B_goal=1.0, type=1):
        '''Default constructor

        :param X_mgap_1:
        The size of the horizontal gap. (entrance)

        :param X_core_1:
            The horizontal coordinate of the end of the iron core (entrance).

        :param X_void_1:
            The horizontal coordinate of the end of the void region (entrance).

        :param X_yoke_1:
            The horizontal coordinate of the end of the magnet end (entrance).

        :param X_mgap_2:
            The size of the horizontal gap. (exit)

        :param X_core_2:
            The horizontal coordinate of the end of the iron core (exit).

        :param X_void_2:
            The horizontal coordinate of the end of the void region (exit).

        :param X_yoke_2:
            The horizontal coordinate of the end of the magnet end (exit).

        :param Y_void_1:
            The vertical coordinate of the end of the void region. (entrance)

        :param Y_yoke_1:
            The vertical coordinate of the end of the iron yoke (entrance).

        :param Y_void_2:
            The vertical coordinate of the end of the void region (exit).

        :param Y_yoke_2:
            The horizontal coordinate of the end of the iron yoke (exit).

        :param Z_len:
            The length of the magnet in the z direction.

        :param Z_pos:
            The position of the magnet in the z direction.

        :param B_goal:
            The desired field in the goal region.

        :param type:
            If 1, the B_goal field is in the core.
            If 2m the B_goal field is in the return yoke. 

        :return:
            Nothing.
        '''
        
        # initialize all the subdomains
        self.domains = []

        points_1 = np.array([[X_mgap_1, 0., Z_pos],
                             [X_core_1, 0., Z_pos],
                             [X_core_1, Y_void_1, Z_pos],
                             [X_mgap_1, Y_yoke_1, Z_pos],
                             [X_mgap_2, 0., Z_pos + Z_len],
                             [X_core_2, 0., Z_pos + Z_len],
                             [X_core_2, Y_void_2, Z_pos + Z_len],
                             [X_mgap_2, Y_yoke_2, Z_pos + Z_len]])
        
        points_2 = np.array([[X_mgap_1, Y_yoke_1, Z_pos],
                             [X_core_1, Y_void_1, Z_pos],
                             [X_void_1, Y_void_1, Z_pos],
                             [X_yoke_1, Y_yoke_1, Z_pos],
                             [X_mgap_2, Y_yoke_2, Z_pos + Z_len],
                             [X_core_2, Y_void_2, Z_pos + Z_len],
                             [X_void_2, Y_void_2, Z_pos + Z_len],
                             [X_yoke_2, Y_yoke_2, Z_pos + Z_len]])

        points_3 = np.array([[X_void_1, 0.0, Z_pos],
                             [X_yoke_1, 0.0, Z_pos],
                             [X_yoke_1, Y_yoke_1, Z_pos],
                             [X_void_1, Y_void_1, Z_pos],
                             [X_void_2, 0.0, Z_pos + Z_len],
                             [X_yoke_2, 0.0, Z_pos + Z_len],
                             [X_yoke_2, Y_yoke_2, Z_pos + Z_len],
                             [X_void_2, Y_void_2, Z_pos + Z_len]]) 
        
        self.domains.append(BrickGeometry(points_1))
        self.domains.append(BrickGeometry(points_2))
        self.domains.append(BrickGeometry(points_3))

        self.B_goal = B_goal

        # compute the areas of the three iron paths
        self.area_1 = 0.5*( (X_core_1 - X_mgap_1)*Z_len  + (X_core_2 - X_mgap_2)*Z_len )
        self.area_2 = 0.5*( (Y_yoke_1 - Y_void_1)*Z_len  + (Y_yoke_2 - Y_void_2)*Z_len )
        self.area_3 = 0.5*( (X_yoke_1 - X_void_1)*Z_len  + (X_yoke_2 - X_void_2)*Z_len )
        
        if type == 1:
            self.Phi = self.B_goal*self.area_1
        
        elif type == 2:
            self.Phi = self.B_goal*self.area_3
        
        else:
            print('type {} unknown!'.format(type))

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # domains[0].plot(ax)
        # domains[1].plot(ax)
        # domains[2].plot(ax)
        # ax.set_aspect('equal')
        # plt.show()

    def compute_B(self, points):
        '''Compute the B field vector inside the brick field magnet
        at certain positions.
        
        :param points:
            The points

        :return:
            The B field vectors.

        '''

        # the number of points
        num_pts = points.shape[0]

        # make a return array
        B = np.zeros((num_pts, 3))

        # loop over the points
        for m, pp in enumerate(points):
            
            # check if it is in domain 1
            if self.domains[0].is_inside(pp):
                B[m, 1] =  self.Phi/self.area_1

            # check if it is in domain 2  
            elif self.domains[1].is_inside(pp):
                B[m, 0] =  self.Phi/self.area_2

            # check if it is in domain 3 
            elif self.domains[2].is_inside(pp):
                B[m, 1] =  -1.0*self.Phi/self.area_3
                
        return B

if __name__ == '__main__':

    # some geometry parameters

    X_mgap_1 = 0.12
    X_core_1 = 0.25
    X_void_1 = 0.41
    X_yoke_1 = 0.66

    X_mgap_2 = 0.2
    X_core_2 = 0.49
    X_void_2 = 0.66
    X_yoke_2 = 1.15

    Y_void_1 = 0.08
    Y_yoke_1 = 0.2
    Y_void_2 = 1.05
    Y_yoke_2 = 1.3

    Z_len = 2.9
    Z_pos = -0.5*Z_len

    bf = BrickField(X_mgap_1, X_core_1, X_void_1, X_yoke_1,
              X_mgap_2, X_core_2, X_void_2, X_yoke_2,
                        Y_void_1, Y_yoke_1,
                        Y_void_2, Y_yoke_2,
                        Z_len, Z_pos=0.0, B_goal=1.7)
    

    # make a meshgrid
    disc_x = 15
    disc_y = 15
    disc_z = 70
    
    X, Y, Z = np.meshgrid(np.linspace(0., 1.5, disc_x),
                          np.linspace(0., 1.5, disc_y),
                          np.linspace(-2.0, 2.0, disc_z))
    
    points = np.zeros((disc_x*disc_y*disc_z, 3))
    points[:, 0] = X.flatten()
    points[:, 1] = Y.flatten()
    points[:, 2] = Z.flatten()
    
    B = bf.compute_B(points)
    
    # the scalars for the arrows
    scalars_field = matlib.repmat(np.linalg.norm(B, axis=1), 15, 1).T.flatten()

    pl = pv.Plotter()

    pl.add_arrows(points, B, mag=0.05, scalars=scalars_field, cmap='jet')

    pl.show()