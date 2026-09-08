from scipy import interpolate
import numpy as np
import pandas as pd

import opensim as osm


class OpenSimModelWrapper:

    def __init__(self, modelPath, motionPath):
        osm.Logger.setLevelString('error')
        self.model = osm.Model(str(modelPath))
        self.model.initSystem()

        # Create time-series table with coordinate values.
        self.table = osm.TimeSeriesTable(str(motionPath))
        tableProcessor = osm.TableProcessor(self.table)
        self.columnLabels = list(self.table.getColumnLabels())
        tableProcessor.append(osm.TabOpUseAbsoluteStateNames())
        self.time = np.asarray(self.table.getIndependentColumn())

        # Convert in radians.
        self.table = tableProcessor.processAndConvertToRadians(self.model)

        # Compute coordinate speeds and accelerations and add speeds to table.
        self.Qs = self.table.getMatrix().to_numpy()
        self.Qds = np.zeros(self.Qs.shape)
        self.Qdds = np.zeros(self.Qs.shape)
        columnAbsoluteLabels = list(self.table.getColumnLabels())
        for i, columnLabel in enumerate(columnAbsoluteLabels):
            spline = interpolate.InterpolatedUnivariateSpline(
                self.time, self.Qs[:,i], k=3)
            # Coordinate speeds
            splineD1 = spline.derivative(n=1)
            self.Qds[:,i] = splineD1(self.time)
            # Coordinate accelerations.
            splineD2 = spline.derivative(n=2)
            self.Qdds[:,i] = splineD2(self.time)
            # Add coordinate speeds to table.
            columnLabel_speed = columnLabel[:-5] + 'speed'
            self.table.appendColumn(
                columnLabel_speed,
                osm.Vector(self.Qds[:,i].flatten().tolist()))

        # Append missing muscle states to table.
        # Needed for StatesTrajectory.
        stateVariableNames = self.model.getStateVariableNames()
        stateVariableNamesStr = [
            stateVariableNames.get(i) for i in range(
                stateVariableNames.getSize())]
        existingLabels = self.table.getColumnLabels()
        for stateVariableNameStr in stateVariableNamesStr:
            if not stateVariableNameStr in existingLabels:
                vec_0 = osm.Vector([0] * self.table.getNumRows())
                self.table.appendColumn(stateVariableNameStr, vec_0)

        # Set state trajectory
        self.stateTrajectory = osm.StatesTrajectory.createFromStatesTable(
            self.model, self.table)

        # Number of muscles.
        self.nMuscles = 0
        self.forceSet = self.model.getForceSet()
        for i in range(self.forceSet.getSize()):
            c_force_elt = self.forceSet.get(i)
            if 'Muscle' in c_force_elt.getConcreteClassName():
                self.nMuscles += 1

        # Coordinates.
        self.coordinateSet = self.model.getCoordinateSet()
        self.nCoordinates = self.coordinateSet.getSize()
        self.coordinates = [self.coordinateSet.get(i).getName()
                            for i in range(self.nCoordinates)]

        # Translational coordinates.
        columnTrLabels = [
            'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        self.idxColumnTrLabels = [
            self.columnLabels.index(i) for i in columnTrLabels]
        self.idxColumnRotLabels = [
            self.columnLabels.index(i) for i in self.columnLabels
            if not i in columnTrLabels]

        self.rootCoordinates = [
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
            'pelvis_tx', 'pelvis_ty', 'pelvis_tz']

        self.lumbarCoordinates = ['lumbar_extension', 'lumbar_bending',
                                  'lumbar_rotation']

        self.armCoordinates = ['arm_flex_r', 'arm_add_r', 'arm_rot_r',
                               'elbow_flex_r', 'pro_sup_r',
                               'arm_flex_l', 'arm_add_l', 'arm_rot_l',
                               'elbow_flex_l', 'pro_sup_l']


    def compute_center_of_mass(self):

        # Compute center of mass position and velocity.
        self.com_values = np.zeros((self.table.getNumRows(),3))
        self.com_speeds = np.zeros((self.table.getNumRows(),3))
        for i in range(self.table.getNumRows()):
            self.model.realizeVelocity(self.stateTrajectory[i])
            self.com_values[i,:] = self.model.calcMassCenterPosition(
                self.stateTrajectory[i]).to_numpy()
            self.com_speeds[i,:] = self.model.calcMassCenterVelocity(
                self.stateTrajectory[i]).to_numpy()

    def get_center_of_mass_values(self):

        self.compute_center_of_mass()
        com_v = self.com_values

        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_v), axis=1)
        columns = ['time'] + ['x','y','z']
        com_values = pd.DataFrame(data=data, columns=columns)

        return com_values

    def get_center_of_mass_speeds(self):

        self.compute_center_of_mass()
        com_s = self.com_speeds

        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_s), axis=1)
        columns = ['time'] + ['x','y','z']
        com_speeds = pd.DataFrame(data=data, columns=columns)

        return com_speeds
