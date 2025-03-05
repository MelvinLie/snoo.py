import numpy as np
import gmsh
import pyvista as pv

def add_SHIP_coil(model, X_core_1, X_void_1, X_yoke_1,
                    X_core_2, X_void_2, X_yoke_2,
                    Z_len, W_coil, H_coil,
                    Z_pos=0.0,
                    show=False,
                    lc=1.0,
                    coil_type=1,
                    sym_xz=True):
    '''Add a coil domain to a gmsh model for SHiP.

    :param model:
        The gmsh model.

    :param X_core_1:
        The horizontal coordinate of the end of the iron core (entrance).

    :param X_void_1:
        The horizontal coordinate of the end of the void region (entrance).

    :param X_yoke_1:
        The horizontal coordinate of the end of the magnet end (entrance).

    :param X_core_2:
        The horizontal coordinate of the end of the iron core (exit).

    :param X_void_2:
        The horizontal coordinate of the end of the void region (exit).

    :param X_yoke_2:
        The horizontal coordinate of the end of the magnet end (exit).

    :param Z_len:
        The length of the magnet in the z direction.

    :param W_coil:
        The coil width.

    :param H_coil:
        The coil height,

    :param Z_pos:
        The position of the magnet in the z direction.

    :param show:
        Set this flag to show the mesh in the gmsh gui.

    :param coil_type:
        The coil type. Type 1 is a coil that is symmetric in the yz plane.

    :param sym_xz:
        Set this flag to true if You want to apply symmetry in the xz plane.

    :return:
        The volume tag of the coil.
    '''

    # the tapering angle alpha
    alpha = np.arctan2(X_core_2 - X_core_1, Z_len)

    # the z position of the entrance
    Z_1 = Z_pos

    # the z position of the exit
    Z_2 = Z_1 + Z_len

    # make the key points
    if sym_xz:
        p1 = model.occ.addPoint(0.0, 0.0, Z_1, lc)
        p2 = model.occ.addPoint(0.0, 0.0, Z_1 - W_coil, lc)
        p3 = model.occ.addPoint(X_core_1, 0.0, Z_1 - W_coil, lc)
        p4 = model.occ.addPoint(X_core_1 + W_coil*np.cos(alpha), 0.0, Z_1 - W_coil*np.sin(alpha), lc)
        p5 = model.occ.addPoint(X_core_2 + W_coil*np.cos(alpha), 0.0, Z_2 +  W_coil*np.sin(alpha), lc)
        p6 = model.occ.addPoint(X_core_2, 0.0, Z_2 + W_coil, lc)
        p7 = model.occ.addPoint(0.0, 0.0, Z_2 + W_coil, lc)
        p8 = model.occ.addPoint(0.0, 0.0, Z_2, lc)
        p9 = model.occ.addPoint(X_core_2, 0.0, Z_2, lc)
        p10 = model.occ.addPoint(X_core_1, 0.0, Z_1, lc)

    else:
        p1 = model.occ.addPoint(0.0, -H_coil , Z_1, lc)
        p2 = model.occ.addPoint(0.0, -H_coil, Z_1 - W_coil, lc)
        p3 = model.occ.addPoint(X_core_1, -H_coil, Z_1 - W_coil, lc)
        p4 = model.occ.addPoint(X_core_1 + W_coil*np.cos(alpha), -H_coil, Z_1 - W_coil*np.sin(alpha), lc)
        p5 = model.occ.addPoint(X_core_2 + W_coil*np.cos(alpha), -H_coil, Z_2 +  W_coil*np.sin(alpha), lc)
        p6 = model.occ.addPoint(X_core_2, -H_coil, Z_2 + W_coil, lc)
        p7 = model.occ.addPoint(0.0, -H_coil, Z_2 + W_coil, lc)
        p8 = model.occ.addPoint(0.0, -H_coil, Z_2, lc)
        p9 = model.occ.addPoint(X_core_2, -H_coil, Z_2, lc)
        p10 = model.occ.addPoint(X_core_1, -H_coil, Z_1, lc)

    # make the lines
    l1 = model.occ.addLine(p1, p2)
    l2 = model.occ.addLine(p2, p3)
    l3 = model.occ.addCircleArc(p3, p10, p4)
    l4 = model.occ.addLine(p4, p5)
    l5 = model.occ.addCircleArc(p5, p9, p6)
    l6 = model.occ.addLine(p6, p7)
    l7 = model.occ.addLine(p7, p8)
    l8 = model.occ.addLine(p8, p9)
    l9 = model.occ.addLine(p9, p10)
    l10 = model.occ.addLine(p10, p1)

    # make a curve loop
    c = model.occ.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10])

    # make the surfaces
    s = model.occ.addPlaneSurface([c])

    # extrude
    if sym_xz:
        extr = model.occ.extrude ([(2, s)], 0., H_coil, 0.)
    else:
        extr = model.occ.extrude ([(2, s)], 0., 2*H_coil, 0.)

    model.occ.synchronize()

    if show:
        gmsh.fltk.run()

    # get the extruded volume
    vol = -1
    for e in extr:
        if e[0] == 3:
            vol = e[1]
            break

    return vol

def add_SHIP_iron_yoke(model, X_mgap_1, X_core_1, X_void_1, X_yoke_1,
                        X_mgap_2, X_core_2, X_void_2, X_yoke_2,
                        Y_core_1, Y_void_1, Y_yoke_1,
                        Y_core_2, Y_void_2, Y_yoke_2,
                        Z_len, 
                        Z_pos=0.0,
                        lc=2e-1,
                        lc_inner=-1,
                        yoke_type=1,
                        reflect_xz=False):
    '''Add an iron yoke for the SHiP project to a gmsh model.
    
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

    :param show:
        Set this flag to show the mesh in the gmsh gui.

    :param lc:
        A length scale parameter to control the mesh size.

    :param lc_inner:
        The legth scale parameter applied in the inner of the iron domain.
        If negative it is ignored.

    :param yoke_type:
        The yoke type. Options are 1, 2, and 3.
        The types 1 and 3 are identical (magnet templates for normal
        conducting magnets)
        The type 2 is the one for the SC magnet.

    :param reflect_xz:
        Reflect the fomain in the xz plane.

    :return:
        The gmsh model object as well as the labels of the physical groups for the two terminals.
    '''

    if lc_inner < 0:
        lc_inner = lc

    if (yoke_type == 1 or yoke_type == 3):

        # the z position of the entrance
        Z_1 = Z_pos

        # the z position of the exit
        Z_2 = Z_1 + Z_len
        
        # THIS IS THE CORE

        # make the points in the xy plane (entrance)
        p1_1 = model.occ.addPoint(X_mgap_1, 0.0, Z_1, lc)
        p2_1 = model.occ.addPoint(X_core_1, 0.0, Z_1, lc)
        p3_1 = model.occ.addPoint(X_core_1, Y_core_1, Z_1, lc)
        p4_1 = model.occ.addPoint(X_void_1, Y_core_1, Z_1, lc)
        p5_1 = model.occ.addPoint(X_void_1, 0.0, Z_1, lc)
        p6_1 = model.occ.addPoint(X_yoke_1, 0.0, Z_1, lc)
        p7_1 = model.occ.addPoint(X_yoke_1, Y_yoke_1, Z_1, lc)
        p8_1 = model.occ.addPoint(X_mgap_1, Y_yoke_1, Z_1, lc)

        # connect the keypoints by lines (entrance)
        l1_1 = model.occ.addLine(p1_1, p2_1)
        l2_1 = model.occ.addLine(p2_1, p3_1)
        l3_1 = model.occ.addLine(p3_1, p4_1)
        l4_1 = model.occ.addLine(p4_1, p5_1)
        l5_1 = model.occ.addLine(p5_1, p6_1)
        l6_1 = model.occ.addLine(p6_1, p7_1)
        l7_1 = model.occ.addLine(p7_1, p8_1)
        l8_1 = model.occ.addLine(p8_1, p1_1)

        # make the points in the xy plane (exit)
        p1_2 = model.occ.addPoint(X_mgap_2, 0.0, Z_2, lc)
        p2_2 = model.occ.addPoint(X_core_2, 0.0, Z_2, lc)
        p3_2 = model.occ.addPoint(X_core_2, Y_core_2, Z_2, lc)
        p4_2 = model.occ.addPoint(X_void_2, Y_core_2, Z_2, lc)
        p5_2 = model.occ.addPoint(X_void_2, 0.0, Z_2, lc)
        p6_2 = model.occ.addPoint(X_yoke_2, 0.0, Z_2, lc)
        p7_2 = model.occ.addPoint(X_yoke_2, Y_yoke_2, Z_2, lc)
        p8_2 = model.occ.addPoint(X_mgap_2, Y_yoke_2, Z_2, lc)

        # connect the keypoints by lines (exit)
        l1_2 = model.occ.addLine(p1_2, p2_2)
        l2_2 = model.occ.addLine(p2_2, p3_2)
        l3_2 = model.occ.addLine(p3_2, p4_2)
        l4_2 = model.occ.addLine(p4_2, p5_2)
        l5_2 = model.occ.addLine(p5_2, p6_2)
        l6_2 = model.occ.addLine(p6_2, p7_2)
        l7_2 = model.occ.addLine(p7_2, p8_2)
        l8_2 = model.occ.addLine(p8_2, p1_2)

        # connect the entrance and exit by lines
        l1_12 = model.occ.addLine(p1_1, p1_2)
        l2_12 = model.occ.addLine(p2_1, p2_2)
        l3_12 = model.occ.addLine(p3_1, p3_2)
        l4_12 = model.occ.addLine(p4_1, p4_2)
        l5_12 = model.occ.addLine(p5_1, p5_2)
        l6_12 = model.occ.addLine(p6_1, p6_2)
        l7_12 = model.occ.addLine(p7_1, p7_2)
        l8_12 = model.occ.addLine(p8_1, p8_2)
        
        # make all curve loops
        c1 = model.occ.addCurveLoop([l1_1, l2_1, l3_1, l4_1, l5_1, l6_1, l7_1, l8_1])
        c2 = model.occ.addCurveLoop([l1_2, l2_2, l3_2, l4_2, l5_2, l6_2, l7_2, l8_2])
        c3 = model.occ.addCurveLoop([l1_1, l2_12, l1_2, l1_12])
        c4 = model.occ.addCurveLoop([l2_1, l3_12, l2_2, l2_12])
        c5 = model.occ.addCurveLoop([l3_1, l4_12, l3_2, l3_12])

        c6 = model.occ.addCurveLoop([l4_1, l5_12, l4_2, l4_12])
        c7 = model.occ.addCurveLoop([l5_1, l6_12, l5_2, l5_12])
        c8 = model.occ.addCurveLoop([l6_1, l7_12, l6_2, l6_12])
        c9 = model.occ.addCurveLoop([l7_1, l8_12, l7_2, l7_12])
        c10 = model.occ.addCurveLoop([l8_1, l1_12, l8_2, l8_12])

        # make the surfaces
        s1 = model.occ.addPlaneSurface([c1])
        s2 = model.occ.addPlaneSurface([c2])
        s3 = model.occ.addPlaneSurface([c3])
        s4 = model.occ.addPlaneSurface([c4])
        s5 = model.occ.addPlaneSurface([c5])
        s6 = model.occ.addPlaneSurface([c6])
        s7 = model.occ.addPlaneSurface([c7])
        s8 = model.occ.addPlaneSurface([c8])
        s9 = model.occ.addPlaneSurface([c9])
        s10 = model.occ.addPlaneSurface([c10])

        # make the iron and air domains
        sl_1 = model.occ.addSurfaceLoop([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10])
        vol_1 = model.occ.addVolume([sl_1])

        model.occ.synchronize()

    elif yoke_type == 2:

        # the z position of the entrance
        Z_1 = Z_pos

        # the z position of the exit
        Z_2 = Z_1 + Z_len
        
        # THIS IS THE CORE

        # make the points in the xy plane (entrance)
        p1_1 = model.occ.addPoint(0.0, Y_void_1, Z_1, lc_inner)
        p2_1 = model.occ.addPoint(X_void_1, Y_void_1, Z_1, lc_inner)
        p3_1 = model.occ.addPoint(X_void_1, 0.0, Z_1, lc_inner)
        p4_1 = model.occ.addPoint(X_yoke_1, 0.0, Z_1, lc)
        p5_1 = model.occ.addPoint(X_yoke_1, Y_yoke_1, Z_1, lc)
        p6_1 = model.occ.addPoint(0.0, Y_yoke_1, Z_1, lc)

        # connect the keypoints by lines (entrance)
        l1_1 = model.occ.addLine(p1_1, p2_1)
        l2_1 = model.occ.addLine(p2_1, p3_1)
        l3_1 = model.occ.addLine(p3_1, p4_1)
        l4_1 = model.occ.addLine(p4_1, p5_1)
        l5_1 = model.occ.addLine(p5_1, p6_1)
        l6_1 = model.occ.addLine(p6_1, p1_1)

        # make the points in the xy plane (exit)
        p1_2 = model.occ.addPoint(0.0, Y_void_2, Z_2, lc_inner)
        p2_2 = model.occ.addPoint(X_void_2, Y_void_2, Z_2, lc_inner)
        p3_2 = model.occ.addPoint(X_void_2, 0.0, Z_2, lc_inner)
        p4_2 = model.occ.addPoint(X_yoke_2, 0.0, Z_2, lc)
        p5_2 = model.occ.addPoint(X_yoke_2, Y_yoke_2, Z_2, lc)
        p6_2 = model.occ.addPoint(0.0, Y_yoke_2, Z_2, lc)

        # connect the keypoints by lines (exit)
        l1_2 = model.occ.addLine(p1_2, p2_2)
        l2_2 = model.occ.addLine(p2_2, p3_2)
        l3_2 = model.occ.addLine(p3_2, p4_2)
        l4_2 = model.occ.addLine(p4_2, p5_2)
        l5_2 = model.occ.addLine(p5_2, p6_2)
        l6_2 = model.occ.addLine(p6_2, p1_2)

        # connect the entrance and exit by lines
        l1_12 = model.occ.addLine(p1_1, p1_2)
        l2_12 = model.occ.addLine(p2_1, p2_2)
        l3_12 = model.occ.addLine(p3_1, p3_2)
        l4_12 = model.occ.addLine(p4_1, p4_2)
        l5_12 = model.occ.addLine(p5_1, p5_2)
        l6_12 = model.occ.addLine(p6_1, p6_2)
        
        # make all curve loops
        c1 = model.occ.addCurveLoop([l1_1, l2_1, l3_1, l4_1, l5_1, l6_1])
        c2 = model.occ.addCurveLoop([l1_2, l2_2, l3_2, l4_2, l5_2, l6_2])
        c3 = model.occ.addCurveLoop([l1_1, l2_12, l1_2, l1_12])
        c4 = model.occ.addCurveLoop([l2_1, l3_12, l2_2, l2_12])
        c5 = model.occ.addCurveLoop([l3_1, l4_12, l3_2, l3_12])

        c6 = model.occ.addCurveLoop([l4_1, l5_12, l4_2, l4_12])
        c7 = model.occ.addCurveLoop([l5_1, l6_12, l5_2, l5_12])

        # make the surfaces
        s1 = model.occ.addPlaneSurface([c1])
        s2 = model.occ.addPlaneSurface([c2])
        s3 = model.occ.addPlaneSurface([c3])
        s4 = model.occ.addPlaneSurface([c4])
        s5 = model.occ.addPlaneSurface([c5])
        s6 = model.occ.addPlaneSurface([c6])
        s7 = model.occ.addPlaneSurface([c7])

        # make the iron and air domains
        sl_1 = model.occ.addSurfaceLoop([s1, s2, s3, s4, s5, s6, s7])
        vol_1 = model.occ.addVolume([sl_1])

        model.occ.synchronize()

    else:
        print('Yoke type {} is unknown!'.format(yoke_type))

    if reflect_xz:

        model.occ.mirror([(3, vol_1)], 0., 1., 0., 0.)
        model.occ.synchronize()
        # dim_tags, map = model.occ.fragment(vol_m, [(3, vol_1)])
        # model.occ.synchronize()

    return vol_1


def add_SHIP_iron_core(model, X_core_1, X_core_2, Y_core_1, Y_core_2,
                        Z_len, Z_pos=0.0, lc=2e-1, sym_xz=True):
    '''Add an iron core for the SHiP project to a gmsh model.
    
    :param X_core_1:
        The horizontal coordinate of the end of the iron core (entrance).

    :param X_core_2:
        The horizontal coordinate of the end of the iron core (exit).

    :param Y_core_1:
        The vertical coordinate of the end of the void region. (entrance)

    :param Y_core_2:
        The horizontal coordinate of the end of the iron yoke (exit).

    :param Z_len:
        The length of the magnet in the z direction.

    :param Z_pos:
        The position of the magnet in the z direction.

    :param show:
        Set this flag to show the mesh in the gmsh gui.

    :param lc:
        A length scale parameter to control the mesh size.

    :param sym_xz:
        Set this flag to true if You want to apply symmetry in the xz plane.
        
    :return:
        The gmsh model object as well as the labels of the physical groups for the two terminals.
    '''

    # the z position of the entrance
    Z_1 = Z_pos

    # the z position of the exit
    Z_2 = Z_1 + Z_len
        
    # THIS IS THE CORE

    if sym_xz:
        # make the points in the xy plane (entrance)
        p1_1 = model.occ.addPoint(0.0, 0.0, Z_1, lc)
        p2_1 = model.occ.addPoint(X_core_1, 0.0, Z_1, lc)
        p3_1 = model.occ.addPoint(X_core_1, Y_core_1, Z_1, lc)
        p4_1 = model.occ.addPoint(0.0, Y_core_1, Z_1, lc)
    else:
        # make the points in the xy plane (entrance)
        p1_1 = model.occ.addPoint(0.0, -Y_core_1, Z_1, lc)
        p2_1 = model.occ.addPoint(X_core_1, -Y_core_1, Z_1, lc)
        p3_1 = model.occ.addPoint(X_core_1, Y_core_1, Z_1, lc)
        p4_1 = model.occ.addPoint(0.0, Y_core_1, Z_1, lc)

    # connect the keypoints by lines (entrance)
    l1_1 = model.occ.addLine(p1_1, p2_1)
    l2_1 = model.occ.addLine(p2_1, p3_1)
    l3_1 = model.occ.addLine(p3_1, p4_1)
    l4_1 = model.occ.addLine(p4_1, p1_1)

    if sym_xz:
        # make the points in the xy plane (exit)
        p1_2 = model.occ.addPoint(0.0, 0.0, Z_2, lc)
        p2_2 = model.occ.addPoint(X_core_2, 0.0, Z_2, lc)
        p3_2 = model.occ.addPoint(X_core_2, Y_core_2, Z_2, lc)
        p4_2 = model.occ.addPoint(0.0, Y_core_2, Z_2, lc)
    else:
        # make the points in the xy plane (exit)
        p1_2 = model.occ.addPoint(0.0, -Y_core_2, Z_2, lc)
        p2_2 = model.occ.addPoint(X_core_2, -Y_core_2, Z_2, lc)
        p3_2 = model.occ.addPoint(X_core_2, Y_core_2, Z_2, lc)
        p4_2 = model.occ.addPoint(0.0, Y_core_2, Z_2, lc)

    # connect the keypoints by lines (exit)
    l1_2 = model.occ.addLine(p1_2, p2_2)
    l2_2 = model.occ.addLine(p2_2, p3_2)
    l3_2 = model.occ.addLine(p3_2, p4_2)
    l4_2 = model.occ.addLine(p4_2, p1_2)
    
    # connect the entrance and exit by lines
    l1_12 = model.occ.addLine(p1_1, p1_2)
    l2_12 = model.occ.addLine(p2_1, p2_2)
    l3_12 = model.occ.addLine(p3_1, p3_2)
    l4_12 = model.occ.addLine(p4_1, p4_2)
        
    # make all curve loops
    c1 = model.occ.addCurveLoop([l1_1, l2_1, l3_1, l4_1])
    c2 = model.occ.addCurveLoop([l1_2, l2_2, l3_2, l4_2])
    c3 = model.occ.addCurveLoop([l1_1, l2_12, l1_2, l1_12])
    c4 = model.occ.addCurveLoop([l2_1, l3_12, l2_2, l2_12])
    c5 = model.occ.addCurveLoop([l3_1, l4_12, l3_2, l3_12])
    c6 = model.occ.addCurveLoop([l4_1, l1_12, l4_2, l4_12])
    
    # make the surfaces
    s1 = model.occ.addPlaneSurface([c1])
    s2 = model.occ.addPlaneSurface([c2])
    s3 = model.occ.addPlaneSurface([c3])
    s4 = model.occ.addPlaneSurface([c4])
    s5 = model.occ.addPlaneSurface([c5])
    s6 = model.occ.addPlaneSurface([c6])
    
    # make the iron and air domains
    sl_1 = model.occ.addSurfaceLoop([s1, s2, s3, s4, s5, s6])
    vol_1 = model.occ.addVolume([sl_1])

    return vol_1

def add_SHIP_iron_yoke_diluted_wing(model, X_mgap_1, X_core_1, X_void_1, X_yoke_1,
                        X_mgap_2, X_core_2, X_void_2, X_yoke_2,
                        Y_core_1, Y_void_1, Y_yoke_1,
                        Y_core_2, Y_void_2, Y_yoke_2,
                        Z_len, 
                        Z_pos=0.0,
                        lc=2e-1,
                        lc_inner=-1,
                        reflect_xz=False):
    '''Add an iron yoke for the SHiP project to a gmsh model. This function is used for the case
    where the wing is diluted.
    
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

    :param show:
        Set this flag to show the mesh in the gmsh gui.

    :param lc:
        A length scale parameter to control the mesh size.

    :param lc_inner:
        The legth scale parameter applied in the inner of the iron domain.
        If negative it is ignored.

    :param reflect_xz:
        Reflect the fomain in the xz plane.

    :return:
        The gmsh model object as well as the labels of the physical groups for the two terminals.
    '''

    if lc_inner < 0:
        lc_inner = lc

    # the z position of the entrance
    Z_1 = Z_pos

    # the z position of the exit
    Z_2 = Z_1 + Z_len

    # THIS IS THE CORE

    # make the points in the xy plane (entrance)
    p1_1 = model.occ.addPoint(X_mgap_1, 0.0, Z_1, lc)
    p2_1 = model.occ.addPoint(X_core_1, 0.0, Z_1, lc)
    p3_1 = model.occ.addPoint(X_core_1, Y_core_1, Z_1, lc)
    p4_1 = model.occ.addPoint(X_yoke_1, Y_core_1, Z_1, lc)
    p5_1 = model.occ.addPoint(X_yoke_1, Y_yoke_1, Z_1, lc)
    p6_1 = model.occ.addPoint(X_mgap_1, Y_yoke_1, Z_1, lc)

    # connect the keypoints by lines (entrance)
    l1_1 = model.occ.addLine(p1_1, p2_1)
    l2_1 = model.occ.addLine(p2_1, p3_1)
    l3_1 = model.occ.addLine(p3_1, p4_1)
    l4_1 = model.occ.addLine(p4_1, p5_1)
    l5_1 = model.occ.addLine(p5_1, p6_1)
    l6_1 = model.occ.addLine(p6_1, p1_1)

    # make the points in the xy plane (exit)
    p1_2 = model.occ.addPoint(X_mgap_2, 0.0, Z_2, lc)
    p2_2 = model.occ.addPoint(X_core_2, 0.0, Z_2, lc)
    p3_2 = model.occ.addPoint(X_core_2, Y_core_2, Z_2, lc)
    p4_2 = model.occ.addPoint(X_yoke_2, Y_core_2, Z_2, lc)
    p5_2 = model.occ.addPoint(X_yoke_2, Y_yoke_2, Z_2, lc)
    p6_2 = model.occ.addPoint(X_mgap_2, Y_yoke_2, Z_2, lc)

    # connect the keypoints by lines (exit)
    l1_2 = model.occ.addLine(p1_2, p2_2)
    l2_2 = model.occ.addLine(p2_2, p3_2)
    l3_2 = model.occ.addLine(p3_2, p4_2)
    l4_2 = model.occ.addLine(p4_2, p5_2)
    l5_2 = model.occ.addLine(p5_2, p6_2)
    l6_2 = model.occ.addLine(p6_2, p1_2)

    # connect the entrance and exit by lines
    l1_12 = model.occ.addLine(p1_1, p1_2)
    l2_12 = model.occ.addLine(p2_1, p2_2)
    l3_12 = model.occ.addLine(p3_1, p3_2)
    l4_12 = model.occ.addLine(p4_1, p4_2)
    l5_12 = model.occ.addLine(p5_1, p5_2)
    l6_12 = model.occ.addLine(p6_1, p6_2)

    # make all curve loops
    c1 = model.occ.addCurveLoop([l1_1, l2_1, l3_1, l4_1, l5_1, l6_1])
    c2 = model.occ.addCurveLoop([l1_2, l2_2, l3_2, l4_2, l5_2, l6_2])

    c3 = model.occ.addCurveLoop([l1_1, l2_12, l1_2, l1_12])
    c4 = model.occ.addCurveLoop([l2_1, l3_12, l2_2, l2_12])
    c5 = model.occ.addCurveLoop([l3_1, l4_12, l3_2, l3_12])
    c6 = model.occ.addCurveLoop([l4_1, l5_12, l4_2, l4_12])
    c7 = model.occ.addCurveLoop([l5_1, l6_12, l5_2, l5_12])
    c8 = model.occ.addCurveLoop([l6_1, l1_12, l6_2, l6_12])

    # make the surfaces
    s1 = model.occ.addPlaneSurface([c1])
    s2 = model.occ.addPlaneSurface([c2])
    s3 = model.occ.addPlaneSurface([c3])
    s4 = model.occ.addPlaneSurface([c4])
    s5 = model.occ.addPlaneSurface([c5])
    s6 = model.occ.addPlaneSurface([c6])
    s7 = model.occ.addPlaneSurface([c7])
    s8 = model.occ.addPlaneSurface([c8])

    # make the iron and air domains
    sl_1 = model.occ.addSurfaceLoop([s1, s2, s3, s4, s5, s6, s7, s8])
    vol_1 = model.occ.addVolume([sl_1])

    model.occ.synchronize()

    # THIS IS THE WING
    pp1_1 = model.occ.addPoint(X_void_1, 0.0, Z_1, lc)
    pp2_1 = model.occ.addPoint(X_yoke_1, 0.0, Z_1, lc)
    pp3_1 = model.occ.addPoint(X_yoke_1, Y_void_1, Z_1, lc)
    pp4_1 = model.occ.addPoint(X_void_1, Y_void_1, Z_1, lc)

    ll1_1 = model.occ.addLine(pp1_1, pp2_1)
    ll2_1 = model.occ.addLine(pp2_1, pp3_1)
    ll3_1 = model.occ.addLine(pp3_1, pp4_1)
    ll4_1 = model.occ.addLine(pp4_1, pp1_1)

    pp1_2 = model.occ.addPoint(X_void_2, 0.0, Z_2, lc)
    pp2_2 = model.occ.addPoint(X_yoke_2, 0.0, Z_2, lc)
    pp3_2 = model.occ.addPoint(X_yoke_2, Y_void_2, Z_2, lc)
    pp4_2 = model.occ.addPoint(X_void_2, Y_void_2, Z_2, lc)

    ll1_2 = model.occ.addLine(pp1_2, pp2_2)
    ll2_2 = model.occ.addLine(pp2_2, pp3_2)
    ll3_2 = model.occ.addLine(pp3_2, pp4_2)
    ll4_2 = model.occ.addLine(pp4_2, pp1_2)

    ll1_12 = model.occ.addLine(pp1_1, pp1_2)
    ll2_12 = model.occ.addLine(pp2_1, pp2_2)
    ll3_12 = model.occ.addLine(pp3_1, pp3_2)
    ll4_12 = model.occ.addLine(pp4_1, pp4_2)

    cc1 = model.occ.addCurveLoop([ll1_1, ll2_1, ll3_1, ll4_1])
    cc2 = model.occ.addCurveLoop([ll1_2, ll2_2, ll3_2, ll4_2])

    cc3 = model.occ.addCurveLoop([ll1_1, ll2_12, ll1_2, ll1_12])
    cc4 = model.occ.addCurveLoop([ll2_1, ll3_12, ll2_2, ll2_12])
    cc5 = model.occ.addCurveLoop([ll3_1, ll4_12, ll3_2, ll3_12])
    cc6 = model.occ.addCurveLoop([ll4_1, ll1_12, ll4_2, ll4_12])
    
    ss1 = model.occ.addPlaneSurface([cc1])
    ss2 = model.occ.addPlaneSurface([cc2])
    ss3 = model.occ.addPlaneSurface([cc3])
    ss4 = model.occ.addPlaneSurface([cc4])
    ss5 = model.occ.addPlaneSurface([cc5])
    ss6 = model.occ.addPlaneSurface([cc6])

    # make the iron and air domains
    sl_2 = model.occ.addSurfaceLoop([ss1, ss2, ss3, ss4, ss5, ss6])
    vol_2 = model.occ.addVolume([sl_2])

    model.occ.synchronize()

    if reflect_xz:

        model.occ.mirror([(3, vol_1), (3, vol_2)], 0., 1., 0., 0.)
        model.occ.synchronize()
        # dim_tags, map = model.occ.fragment(vol_m, [(3, vol_1)])
        # model.occ.synchronize()

    return vol_1, vol_2

def add_SHiP_volume(model, points_1, points_2):
    '''Add a volume based on points at entrance and exit.
    The points are connected sequentially by lines. The order of the
    points in the input lists is followed.

    :param model:
        The gmsh model.
    
    :param points_1:
        A list of gmsh points at the entrance of the magnet.

    :param points_2:
        A list of gmsh points at the exit of the magnet.

    :return:
        The gmsh volume tag.
    '''

    # the number of points
    num_points = len(points_1)

    # connect the entrance points by lines
    l_1 = [model.occ.addLine(points_1[i], points_1[i+1]) for i in range(num_points-1)]
    l_1.append(model.occ.addLine(points_1[-1], points_1[0]))

    # connect the exit points by lines
    l_2 = [model.occ.addLine(points_2[i], points_2[i+1]) for i in range(num_points-1)]
    l_2.append(model.occ.addLine(points_2[-1], points_2[0]))

    # connect entrance and exit by lines
    l_12 = [model.occ.addLine(points_1[i], points_2[i]) for i in range(num_points)]

    # make curve loops for entrance and exit
    c_1 = model.occ.addCurveLoop(l_1)
    c_2 = model.occ.addCurveLoop(l_2)

    # make curve loops connecting entrance and exit
    c_12 = [model.occ.addCurveLoop([l_1[i], l_12[i+1], l_2[i], l_12[i]])
                                    for i in range(num_points-1)]
    c_12.append(model.occ.addCurveLoop([l_1[-1], l_12[0], l_2[-1], l_12[-1]]))

    # add plane surfaces
    s = [model.occ.addPlaneSurface([c_1]),
         model.occ.addPlaneSurface([c_2])]
    for i, cc in enumerate(c_12):
        s.append(model.occ.addPlaneSurface([cc]))

    # make surface loop
    sl = model.occ.addSurfaceLoop(s)

    # make the volume
    vol = model.occ.addVolume([sl])

    return vol

def add_SHIP_iron_yoke_diluted_core(model, X_mgap_1, X_core_1, X_void_1, X_yoke_1,
                        X_mgap_2, X_core_2, X_void_2, X_yoke_2,
                        Y_core_1, Y_void_1, Y_yoke_1,
                        Y_core_2, Y_void_2, Y_yoke_2,
                        Z_len, 
                        Z_pos=0.0,
                        lc=2e-1,
                        lc_inner=-1,
                        reflect_xz=False):
    '''Add an iron yoke for the SHiP project to a gmsh model. This function is used for the case
    where the core is diluted.
    
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

    :param show:
        Set this flag to show the mesh in the gmsh gui.

    :param lc:
        A length scale parameter to control the mesh size.

    :param lc_inner:
        The legth scale parameter applied in the inner of the iron domain.
        If negative it is ignored.

    :param reflect_xz:
        Reflect the fomain in the xz plane.

    :return:
        The gmsh model object as well as the labels of the physical groups for the two terminals.
    '''

    if lc_inner < 0:
        lc_inner = lc

    # the z position of the entrance
    Z_1 = Z_pos

    # the z position of the exit
    Z_2 = Z_1 + Z_len

    # THESE ARE THE KEYPOINTS FOR THE CORE
    p_core_1 = [model.occ.addPoint(X_mgap_1, 0.0, Z_1, lc_inner),
              model.occ.addPoint(X_core_1, 0.0, Z_1, lc_inner),
              model.occ.addPoint(X_core_1, Y_core_1, Z_1, lc_inner),
              model.occ.addPoint(0.0, Y_core_1, Z_1, lc_inner)]

    p_core_2 = [model.occ.addPoint(X_mgap_2, 0.0, Z_2, lc_inner),
              model.occ.addPoint(X_core_2, 0.0, Z_2, lc_inner),
              model.occ.addPoint(X_core_2, Y_core_2, Z_2, lc_inner),
              model.occ.addPoint(0.0, Y_core_2, Z_2, lc_inner)]

    # THESE ARE THE KEYPOINTS FOR THE RETURN YOKE
    p_yoke_1 = [model.occ.addPoint(X_void_1, 0.0, Z_1, lc),
              model.occ.addPoint(X_yoke_1, 0.0, Z_1, lc),
              model.occ.addPoint(X_yoke_1, Y_yoke_1, Z_1, lc),
              model.occ.addPoint(0.0, Y_yoke_1, Z_1, lc),
              model.occ.addPoint(0.0, Y_core_1, Z_1, lc),
              model.occ.addPoint(X_void_1, Y_core_1, Z_1, lc)]

    p_yoke_2 = [model.occ.addPoint(X_void_2, 0.0, Z_2, lc),
              model.occ.addPoint(X_yoke_2, 0.0, Z_2, lc),
              model.occ.addPoint(X_yoke_2, Y_yoke_2, Z_2, lc),
              model.occ.addPoint(0.0, Y_yoke_2, Z_2, lc),
              model.occ.addPoint(0.0, Y_core_2, Z_2, lc),
              model.occ.addPoint(X_void_2, Y_core_2, Z_2, lc)]

    vol_core = add_SHiP_volume(model, p_core_1, p_core_2)
    vol_yoke = add_SHiP_volume(model, p_yoke_1, p_yoke_2)

    model.occ.synchronize()

    if reflect_xz:

        model.occ.mirror([(3, vol_core), (3, vol_yoke)], 0., 1., 0., 0.)
        model.occ.synchronize()
        # dim_tags, map = model.occ.fragment(vol_m, [(3, vol_1)])
        # model.occ.synchronize()

    return vol_core, vol_yoke

def add_SHIP_iron_yoke_mag_4(model, X_A_1, X_B_1, X_C_1, X_D_1, X_E_1, X_F_1,
                        X_A_2, X_B_2, X_C_2, X_D_2, X_E_2, X_F_2,
                        Y_core_1, Y_yoke_1,
                        Y_core_2, Y_yoke_2,
                        Z_len, 
                        Z_pos=0.0,
                        lc=2e-1,
                        lc_inner=-1,
                        reflect_xz=False):
    '''Add an iron yoke for the SHiP project to a gmsh model. This function is used for the Mag4 template
    
    :param X_A_1:
        The the horizontal coordinate of the first keypoint (entrance).

    :param X_B_1:
        The the horizontal coordinate of the second keypoint (entrance).

    :param X_C_1:
        The the horizontal coordinate of the third keypoint (entrance).

    :param X_D_1:
        The the horizontal coordinate of the fourth keypoint (entrance).

    :param X_E_1:
        The the horizontal coordinate of the fifth keypoint (entrance).

    :param X_F_1:
        The the horizontal coordinate of the sixths keypoint (entrance).

    :param X_yoke_1:
        The horizontal coordinate of the end of the magnet end (entrance).

    :param X_A_2:
        The the horizontal coordinate of the first keypoint (exit).

    :param X_B_2:
        The the horizontal coordinate of the second keypoint (exit).

    :param X_C_2:
        The the horizontal coordinate of the third keypoint (exit).

    :param X_D_2:
        The the horizontal coordinate of the fourth keypoint (exit).

    :param X_E_2:
        The the horizontal coordinate of the fifth keypoint (exit).

    :param X_F_2:
        The the horizontal coordinate of the sixths keypoint (exit).

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

    :param show:
        Set this flag to show the mesh in the gmsh gui.

    :param lc:
        A length scale parameter to control the mesh size.

    :param lc_inner:
        The legth scale parameter applied in the inner of the iron domain.
        If negative it is ignored.

    :param reflect_xz:
        Reflect the fomain in the xz plane.

    :return:
        The gmsh model object as well as the labels of the physical groups for the two terminals.
    '''

    if lc_inner < 0:
        lc_inner = lc

    # the z position of the entrance
    Z_1 = Z_pos

    # the z position of the exit
    Z_2 = Z_1 + Z_len

    # THESE ARE THE KEYPOINTS FOR THE CORE
    p_core_1 = [model.occ.addPoint(X_C_1, 0.0, Z_1, lc_inner),
              model.occ.addPoint(X_D_1, 0.0, Z_1, lc_inner),
              model.occ.addPoint(X_D_1, Y_core_1, Z_1, lc_inner),
              model.occ.addPoint(X_C_1, Y_core_1, Z_1, lc_inner)]

    p_core_2 = [model.occ.addPoint(X_C_2, 0.0, Z_2, lc_inner),
              model.occ.addPoint(X_D_2, 0.0, Z_2, lc_inner),
              model.occ.addPoint(X_D_2, Y_core_2, Z_2, lc_inner),
              model.occ.addPoint(X_C_2, Y_core_2, Z_2, lc_inner)]

    # THESE ARE THE KEYPOINTS FOR THE RETURN YOKE
    p_yoke_1 = [model.occ.addPoint(X_A_1, 0.0, Z_1, lc),
              model.occ.addPoint(X_B_1, 0.0, Z_1, lc),
              model.occ.addPoint(X_B_1, Y_core_1, Z_1, lc),
              model.occ.addPoint(X_E_1, Y_core_1, Z_1, lc),
              model.occ.addPoint(X_E_1, 0.0, Z_1, lc),
              model.occ.addPoint(X_F_1, 0.0, Z_1, lc),
              model.occ.addPoint(X_F_1, Y_yoke_1, Z_1, lc),
              model.occ.addPoint(X_A_1, Y_yoke_1, Z_1, lc)]

    p_yoke_2 = [model.occ.addPoint(X_A_2, 0.0, Z_2, lc),
              model.occ.addPoint(X_B_2, 0.0, Z_2, lc),
              model.occ.addPoint(X_B_2, Y_core_2, Z_2, lc),
              model.occ.addPoint(X_E_2, Y_core_2, Z_2, lc),
              model.occ.addPoint(X_E_2, 0.0, Z_2, lc),
              model.occ.addPoint(X_F_2, 0.0, Z_2, lc),
              model.occ.addPoint(X_F_2, Y_yoke_2, Z_2, lc),
              model.occ.addPoint(X_A_2, Y_yoke_2, Z_2, lc)]

    vol_core = add_SHiP_volume(model, p_core_1, p_core_2)
    vol_yoke = add_SHiP_volume(model, p_yoke_1, p_yoke_2)

    model.occ.synchronize()

    if reflect_xz:

        model.occ.mirror([(3, vol_core), (3, vol_yoke)], 0., 1., 0., 0.)
        model.occ.synchronize()
        # dim_tags, map = model.occ.fragment(vol_m, [(3, vol_1)])
        # model.occ.synchronize()

    return vol_core, vol_yoke

def compute_area_polygon(kp, isclosed=True):
    '''Compute the area of a non-intersecting (default = closed) polygon.
    Closed means that the first and last keypoints is identical. If this
    is not the case, set the isclosed flag to false.
    
    :param kp:
        The keypoints in an array of size (M x 2).
        
    :param isclosed:
        Set this flag if the polygon is not closed, that means if first and last
        keypoint are not identical.

    :return:
        The area.
        
    '''
    area = 0.0

    for i in range(kp.shape[0]-1):
        area += kp[i, 0]*kp[i+1, 1] - kp[i+1, 0]*kp[i, 1]

    if not isclosed:
        area += kp[-1, 0]*kp[0, 1] - kp[0, 0]*kp[-1, 1]

    return 0.5*area