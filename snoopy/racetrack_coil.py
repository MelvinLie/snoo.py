import numpy as np
import gmsh
from tqdm import tqdm
from scipy.sparse import csr_array
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import BSpline
import pyvista as pv

from . import fem_c_mod as fem_c



def compute_2D_line_intersection(intercept_1, slope_1, intercept_2, slope_2):
    '''Compute the intersection point of two lines in 2D.

    :param intercept_1:
        The two coordinates of the interception point of line 1.

    :param slope_1:
        The two coordinates of the slope of line 1.

    :param intercept_2:
        The two coordinates of the interception point of line 2.

    :param slope_2:
        The two coordinates of the slope of line 2.
        
    :return:
        The two coordinates of the interception point.
    '''
    M = np.array([[slope_1[0] , -slope_2[0]],
                  [slope_1[1], -slope_2[1]]])

    b = np.array([intercept_2[0] - intercept_1[0],
                  intercept_2[1] - intercept_1[1]])

    x_2 = np.linalg.solve(M, b)

    return slope_1*x_2[0] + intercept_1


class RacetrackCoil():

    def __init__(self, kp, y, rad, current=1.0):
        '''Default constructor.
        
        :param kp:
            The 8 keypoints in the x,z plane.

        :param y:
            An array with vertical coordinates. There is one turn on
            each vertical coordinate.

        :param rad:
            The coil radius.

        :param current:
            The coil current.

        :return:
            None.
        '''

        self.kp = kp
        self.y = y
        self.rad = rad
        self.current = current

        return None

    def compute_B(self, positions, disc_arc=5):
        '''Compute the B field at given positions.
        
        :param positions:
            The points to evaluate.
            
        :param disc_arc:
            The discretization of the arcs into this number of segments.
            Default = 5.
            
        :return:
            The B field vectors.
            
        '''
        segs = self.get_segments(disc_arc=disc_arc)

        return fem_c.compute_B_line_segs_cpp(segs, positions, self.current, self.rad)

    def get_segments(self, disc_arc=5, enable_plot=False):
        '''Get the segments for this racetrack wire
        
        :param disc_arc:
            Optional parameter. The arcs are discretized according to this
            setting.

        :param enable_plot:
            Optional parameter. To enable a plot.

        :return:
            An (M x 6) numpy array where M is the number of segments.
        '''

        # we will use quadratic BSplines for the arcs
        k = 2

        # this is the knot vector
        t = [0, 0, 0, 1, 1, 1]

        # these will be the coefficients
        cx = [0., 0., 0.]
        cz = [0., 0., 0.]

        # this is a parameter vector we use to discretize the splines
        t_vec = np.linspace(0., 1., disc_arc+1)

        # on each level we have 4*disc_arc + 4 segments
        segs_per_lvl = 4*disc_arc + 4

        # make the argments on the xz plane
        segs_base = np.zeros((segs_per_lvl, 6))

        # this is the 'front' segment
        segs_base[0, :3] = np.array([self.kp[7, 0], 0.0, self.kp[7, 1]])
        segs_base[0, 3:] = np.array([self.kp[0, 0], 0.0, self.kp[0, 1]])

        # we determine the control point of the first arc from the intersection
        # of two lines       
        intersect_1 = compute_2D_line_intersection(self.kp[7, :],
                                                    self.kp[0, :] - self.kp[7, :],
                                                    self.kp[1, :],
                                                    self.kp[2, :] - self.kp[1, :])

        # make the first arc
        cx = [self.kp[0, 0], intersect_1[0], self.kp[1, 0]]
        cz = [self.kp[0, 1], intersect_1[1], self.kp[1, 1]]
        
        spl_x = BSpline(t, cx, k)(t_vec)
        spl_z = BSpline(t, cz, k)(t_vec)

        for i in range(disc_arc):
            segs_base[1+i, :3] = np.array([spl_x[i], 0.0, spl_z[i]])
            segs_base[1+i, 3:] = np.array([spl_x[i+1], 0.0, spl_z[i+1]])

        # this is the next straight segment
        segs_base[disc_arc+1, :3] = np.array([self.kp[1, 0], 0.0, self.kp[1, 1]])
        segs_base[disc_arc+1, 3:] = np.array([self.kp[2, 0], 0.0, self.kp[2, 1]])


        # we determine the control point of the first arc from the intersection
        # of two lines
        intersect_2 = compute_2D_line_intersection(self.kp[1, :],
                                                    self.kp[2, :] - self.kp[1, :],
                                                    self.kp[3, :],
                                                    self.kp[4, :] - self.kp[3, :])

        # make the first arc
        cx = [self.kp[2, 0], intersect_2[0], self.kp[3, 0]]
        cz = [self.kp[2, 1], intersect_2[1], self.kp[3, 1]]
        
        spl_x = BSpline(t, cx, k)(t_vec)
        spl_z = BSpline(t, cz, k)(t_vec)

        for i in range(disc_arc):
            segs_base[2+disc_arc+i, :3] = np.array([spl_x[i], 0.0, spl_z[i]])
            segs_base[2+disc_arc+i, 3:] = np.array([spl_x[i+1], 0.0, spl_z[i+1]])

        # this is the next straight segment
        segs_base[2+2*disc_arc, :3] = np.array([self.kp[3, 0], 0.0, self.kp[3, 1]])
        segs_base[2+2*disc_arc, 3:] = np.array([self.kp[4, 0], 0.0, self.kp[4, 1]])

        # we determine the control point of the first arc from the intersection
        # of two lines
        intersect_3 = compute_2D_line_intersection(self.kp[3, :],
                                                    self.kp[4, :] - self.kp[3, :],
                                                    self.kp[5, :],
                                                    self.kp[6, :] - self.kp[5, :])

        # make the first arc
        cx = [self.kp[4, 0], intersect_3[0], self.kp[5, 0]]
        cz = [self.kp[4, 1], intersect_3[1], self.kp[5, 1]]
        
        spl_x = BSpline(t, cx, k)(t_vec)
        spl_z = BSpline(t, cz, k)(t_vec)

        for i in range(disc_arc):
            segs_base[3+2*disc_arc+i, :3] = np.array([spl_x[i], 0.0, spl_z[i]])
            segs_base[3+2*disc_arc+i, 3:] = np.array([spl_x[i+1], 0.0, spl_z[i+1]])

        # this is the next straight segment
        segs_base[3+3*disc_arc, :3] = np.array([self.kp[5, 0], 0.0, self.kp[5, 1]])
        segs_base[3+3*disc_arc, 3:] = np.array([self.kp[6, 0], 0.0, self.kp[6, 1]])

        # we determine the control point of the first arc from the intersection
        # of two lines
        intersect_4 = compute_2D_line_intersection(self.kp[5, :],
                                                    self.kp[6, :] - self.kp[5, :],
                                                    self.kp[7, :],
                                                    self.kp[0, :] - self.kp[7, :])

        # make the first arc
        cx = [self.kp[6, 0], intersect_4[0], self.kp[7, 0]]
        cz = [self.kp[6, 1], intersect_4[1], self.kp[7, 1]]
        
        spl_x = BSpline(t, cx, k)(t_vec)
        spl_z = BSpline(t, cz, k)(t_vec)

        for i in range(disc_arc):
            segs_base[4+3*disc_arc+i, :3] = np.array([spl_x[i], 0.0, spl_z[i]])
            segs_base[4+3*disc_arc+i, 3:] = np.array([spl_x[i+1], 0.0, spl_z[i+1]])

        # the return data
        segs_ret = np.zeros((0, 6))

        # now stack vertically
        for i, yy in enumerate(self.y):
            segs_base[:, 1] = yy
            segs_base[:, 4] = yy
            segs_ret = np.append(segs_ret, segs_base, axis=0)

        if enable_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            for i, seg in enumerate(segs_ret):
                ax.plot([seg[0], seg[3]],
                        [seg[1], seg[4]],
                        [seg[2], seg[5]])
            
            for i, yy in enumerate(self.y):
                ax.plot(intersect_1[0], yy, intersect_1[1], 'o', color='red')
                ax.plot(intersect_2[0], yy, intersect_2[1], 'o', color='red')
                ax.plot(intersect_3[0], yy, intersect_3[1], 'o', color='red')
                ax.plot(intersect_4[0], yy, intersect_4[1], 'o', color='red')
            ax.set_xlabel('$x$ in m')
            ax.set_xlabel('$y$ in m')
            ax.set_xlabel('$z$ in m')
            ax.set_aspect('equal')
            
            plt.show()

        return segs_ret
    

    def plot_mpl(self, ax):
        '''Plot the geometry in a matplotlib axes object.

        :param ax:
            A matplotlib axes object.

        :return:
            None.
        '''

        segs = self.get_segments()
            
        for i, seg in enumerate(segs):
            ax.plot([seg[0], seg[3]],
                    [seg[1], seg[4]],
                    [seg[2], seg[5]])
            
        return
    
    def plot_pv(self, pl, color=[255/255, 40/255, 0/255], width=8):
        '''Plot the geometry in a pyvista plotter object.

        :param pl:
            A pyvista plotter object.

        :return:
            None.
        '''

        segs = self.get_segments()

        points, cells = self.make_surface_mesh(disc_arc=5)


        # make the mesh object
        cell_info = np.append(np.ones((len(cells), 1), dtype=np.int64)*4, np.int64(cells), axis=1)
        mesh = pv.UnstructuredGrid(cell_info, [pv.CellType.QUAD]*cells.shape[0], points)

        # pl.add_mesh(mesh, color=color)
        surf = mesh.extract_surface()
        surf.compute_normals(inplace=True, split_vertices=True)
        pl.add_mesh(surf, color=color, pbr=True, metallic=0.5, roughness=0.3)

        
        return

    def make_surface_mesh(self, resol=30, disc_arc=5):
        '''Mesh the surface of the cable.

        :param resol:
            The discretization in the azimutal direction. Default 12.

        :return:
            The nodes and the connectivity of the mesh.
        '''

        # get the segments
        segs = self.get_segments(disc_arc=disc_arc)

        # the number of segments
        num_seg = segs.shape[0]

        # number of vertical positions
        num_y = len(self.y)

        # the number of segments per level
        num_segs_per_level = np.int32(num_seg/num_y)

        # total number of points
        num_points = (num_segs_per_level+1)*resol

        # points container
        points = np.zeros((num_points, 3))

        # number of cells
        num_cells = num_segs_per_level*(resol-1)

        # cells container
        cells = np.zeros((num_cells, 4), dtype=np.int32)

        # the azimuth angles
        phi = np.linspace(-np.pi, np.pi, resol)

        # fill the points
        for i, seg in enumerate(segs[:num_segs_per_level, :]):

            # the direction vector
            d = (seg[3:] - seg[:3])/np.linalg.norm(seg[3:] - seg[:3])

            # the transversal unit vectors
            e_u = np.cross(d, np.array([0., 1., 0.]))

            for j, pphi in enumerate(phi):

                points[i*resol + j, 0] = seg[0] + self.rad*np.cos(pphi)*e_u[0]
                points[i*resol + j, 1] = seg[1] + self.rad*np.sin(pphi)
                points[i*resol + j, 2] = seg[2] + self.rad*np.cos(pphi)*e_u[2]

            if i == num_segs_per_level - 1:
                for j, pphi in enumerate(phi):

                    points[(i+1)*resol + j, 0] = seg[3] + self.rad*np.cos(pphi)*e_u[0]
                    points[(i+1)*resol + j, 1] = seg[4] + self.rad*np.sin(pphi)
                    points[(i+1)*resol + j, 2] = seg[5] + self.rad*np.cos(pphi)*e_u[2]



        # fill the cells
        for i in range(num_segs_per_level):
            for j in range(resol - 1):
                cells[i*(resol-1) + j, :] = np.array([i*resol + j,
                                                     (i+1)*resol + j,
                                                      (i+1)*resol + j + 1,
                                                       i*resol + j + 1], dtype=np.int32)

            
        # all points
        all_points = np.zeros((len(self.y)*points.shape[0], 3))

        # all cells
        all_cells = np.zeros((len(self.y)*cells.shape[0], 4), dtype=np.int32)

        # stack the coils again
        for i, yy in enumerate(self.y):
            
            all_points[i*points.shape[0]:(i+1)*points.shape[0], :] = points
            all_points[i*points.shape[0]:(i+1)*points.shape[0], 1] += yy - self.y[0]

            all_cells[i*cells.shape[0]:(i+1)*cells.shape[0], :] = cells + i*points.shape[0]

        return all_points, all_cells
