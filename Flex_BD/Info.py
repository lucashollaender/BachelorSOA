""" ------ File Setup ------ """
# Remember to run this:
# from <SOALIB.>RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation

""" ------ Body Setup ------ """
# *** Body Parameters ****
# klOO:     Hinge position (row vector)
# H_type:   Hinge type (string)
# m:        Mass (scalar)
# CkJk:     Inertia (row vector)
# klOC:     COM position (row vector)

# *** Create Body ***
# joint = Joint(<klOO>, <H_type>)
# inertia = Inertia(<m>, <CkJk>, <klOC>)
# body = SOABody(<joint>, <inertia>)

""" ------ Body Attributes ------ """
# If not specified program assumes zero column vectors
# theta0, beta0, tau, F_ext ---> column vectors

# *** Initial condition ***
# body.set_initial_theta0(<theta0>)   //   <theta0> ---> column vector
#       --->   "revx/y/z" use: theta0 = np.deg2rad(theta_x/y/z)
#       --->   "spherical" use: theta0 = q0 = sb.get_quat_from_degrees(theta_x, theta_y, theta_z)
#       --->   "free" use: theta0 = np.vstack([q0, l]), where l is the initial linear displacement (l = [l_x, l_y, l_z])
#       --->   "fixed" use: theta0 cannot be specified
# body.set_initial_beta0(<beta0>)   //   <beta0> ---> column vector
#       --->   "revx/y/z" use: beta0 = omega_x/y/z
#       --->   "spherical" use: beta0 = np.array([omega_x, omega_y, omega_z]).reshape(3, 1)
#       --->   "free" use: beta0 = np.array([omega_x, omega_y, omega_z, v_x, v_y, v_z]), where v is the initial linear velocity (v = [v_x, v_y, v_z])
#       --->   "fixed" use: beta0 cannot be specified

# *** Forces ***
# body.set_tau(<tau>)   //   <tau> ---> column vector, np.array([<tau>]).reshape(nDOF, 1)

# body.set_F_ext(<F_ext>, <klBO>)   //   <F_ext>, <klBO> ---> lists of same length
#       --->   F_ext is a list of column vectors (6, 1) with external forces
#       --->   klBO is a list of row vectors (1, 3) with the external forces' appliying position

""" ------ System Setup and Simulation ------ """
# *** Multibody System ***
# system = MultibodySystem(bodies)
#       --->   bodies = [body_1, body_2, ..., body_n], list of bodies created above (tip: b_1 and base: b_n)

# *** Simulation Setup ***
# sim = Simulation(system, tf, dt)
#       --->   system as created above
#       --->   tf, length of simulation
#       --->   dt, time step size

""" ------ Camera Settings ------ """
# sim.camera_speed(x)
#       --->   x, number from -1..1 defining the speed in both directions (zero if not changed)

# sim.camera_hor(x)
#       --->   x, number from 0..360 defining the camera rotation around z-axis (zero if not changed)

# sim.camera_ver(x)
#       --->   x, number from -90..90 defining the camera rotation around x-axis (20 if not changed)

""" ------ Parameter Call ------ """
# *** Parameter Call ***   //   Get parameter for each body for each time step
# sim.get_<parameter>()
#       --->   <parameter> = [state, X, pos, V, alpha, a, b]

""" ------ Animate or Render Animation ------ """
# *** Animate ***
# sim.animate()

# *** Render Animation to HTML-file ***   // <file_name>, r<file_path> ---> strings
# sim.animate(<file_name>, <file_path>)
#       --->   <file_name>, name of the render file
#       --->   <file_path>, copy the file path of the folder you want to save the HTML-file in.
#               nb! You must add the letter <r> in front of file path string as:
#               file_path = r"C:\Users\jepp6\OneDrive..."
#               Choose another folder than the GIT-Hub synchronize folder, since the file will
#               be to big and result in a "commit" error.
