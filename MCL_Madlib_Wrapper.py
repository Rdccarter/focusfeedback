
# use ctypes for easy access to dll
from ctypes import *
import os

class MCL_Nanodrive:
    
    def __init__(self):

        os.add_dll_directory(os.getcwd())

        #load the dll
        self.dll = cdll.Madlib

        #Handle Management
        self.dll.MCL_ReleaseHandle.restype = None
        self.dll.MCL_ReleaseAllHandles.restype = None

        #Standard Device Movement
        self.dll.MCL_SingleReadZ.restype = c_double
        self.dll.MCL_SingleReadN.restype = c_double
        self.dll.MCL_MonitorZ.restype = c_double
        self.dll.MCL_MonitorN.restype = c_double

        #Encoder
        self.dll.MCL_ReadEncoderZ.restype = c_double

        #Tip Tilt Z
        self.dll.MCL_TipTiltHeight.restype = c_double
        self.dll.MCL_TipTiltWidth.restype = c_double
        self.dll.MCL_GetTipTiltThetaY.restype = c_double
        self.dll.MCL_GetTipTiltCenter.restype = c_double
        self.dll.MCL_GetTipTiltThetaX.restype = c_double

        #Device Information
        self.dll.MCL_DeviceAttached.restype = c_bool
        self.dll.MCL_GetCalibration.restype = c_double
        self.dll.MCL_PrintDeviceInfo.restype = None
        self.dll.MCL_DLLVersion.restype = None
        self.dll.MCL_CorrectDriverVersion = c_bool

        #Variable for Storing Waveform Array Length
        self.__wfma_len = None

        #Error Codes
        self.error_list = [-1, -2, -3, -4, -5, -6, -7, -8]

    # Handle Management

    def init_handle(self):
        """Requests control of a single Mad City Labs Nano-Drive.

        Returns:
            Returns a valid handle (int).

        Raises:
            Raises MCL Exception (int).
        """
        err = self.dll.MCL_InitHandle()
        if err == 0:
            raise MCL_ND_Exceptions(-8)
        return err

    def init_handle_or_get_existing(self):
        """Requests control of a Nano-Drive or gets a handle to a currently
        controlled Nano-Drive
        
        Request control of a single Mad City Labs Nano-Drive. If all attached 
        Nano-Drives are controlled, this function will return a handle to one
        of the Nano-Drives currently controlled by the DLL.  

        Returns:
            Returns a valid handle

        Raises:
            Raises MCL Exception
        """
        err = self.dll.MCL_InitHandleOrGetExisting()
        if err == 0:
            raise MCL_ND_Exceptions(-8)
        return err

    def grab_handle(self, device_id):
        """Requests control of a specific type of Mad City Labs Nano-Drive.

        Args:
            device_id:
                Nano-Drive Single Axis          8193    0x2001
                Nano-Drive Three Axis           8195    0x2003
                Nano-Drive Four Axis            8196    0x2004
                Nano-Drive 16 bit Tip/Tilt Z    8275    0x2053
                Nano-Drive 20 bit Single Axis   8705    0x2201
                Nano-Drive 20 bit Three Axis    8707    0x2203
                Nano-Drive 20 bit Tip/Tilt Z    8787    0x2253
                Nano-Gauge                      8448    0x2100
                C-Focus                         9217    0x2401

        Returns:
            Returns a valid handle (int).

        Raises:
            Raises MCL Exception
        """
        err = self.dll.MCL_GrabHandle(c_ushort(device_id))
        if err == 0:
            raise MCL_ND_Exceptions(-8)
        return err

    def grab_handle_or_get_existing(self, device_id):
        """Uses device_id to grab a handle of a specific Nano-Drive device

        Requests control of a specific type of Mad City Labs device. If all 
        attached Nano-Drives of the specified type are controlled, this function
        will return a handle to one of the Nano-Drives of that type currently
        controlled by the DLL. 

        Args:
            device_id:
                Nano-Drive Single Axis          8193    0x2001
                Nano-Drive Three Axis           8195    0x2003
                Nano-Drive Four Axis            8196    0x2004
                Nano-Drive 16 bit Tip/Tilt Z    8275    0x2053
                Nano-Drive 20 bit Single Axis   8705    0x2201
                Nano-Drive 20 bit Three Axis    8707    0x2203
                Nano-Drive 20 bit Tip/Tilt Z    8787    0x2253
                Nano-Gauge                      8448    0x2100
                C-Focus                         9217    0x2401

        Returns:
            Returns a valid handle (int).

        Raises:
            Raises MCL Exception
        """
        err = self.dll.MCL_GrabHandleOrGetExisting(c_ushort(device_id))
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return err

    def grab_all_handles(self):
        """
        Requests control of all of the attached Mad City Labs Nano-Drives that 
        are not yet under control.

        Returns:
            Returns the number of Nano-Drives currently controlled by this 
            instance of the DLL (int).
        """
        return self.dll.MCL_GrabAllHandles()

    def get_all_handles(self, size):
        """
        Fills a list with valid handles to the Nano-Drives currently under the 
        control of this instance of the DLL.

        Args:
            size (int): Size of the 'handles' array

        Returns:
            Returns the number of valid handles in the 'handles' array (int).
            Returns list of handles (ints list).
        """
        handles_list = (c_int32 * size)()
        num_handles = self.dll.MCL_GetAllHandles(pointer(handles_list), size)
        return num_handles.value, handles_list

    def number_of_current_handles(self):
        """
        Returns the number of Nano-Drives currently controlled by this instance 
        of the DLL (int).

        Returns:
            Number of Nano-Drives controlled (int).
        """
        return self.dll.MCL_NumberOfCurrentHandles();

    def get_handle_by_serial(self, serial_num):
        """Searches for a Nano-Drive with serial: serial_num

        Searches Nano-Drives currently controlled for a Nano-Drive whose serial
        number matches 'serial_num'. Since this function only searches through
        Nano-Drives which the DLL is controlling, grab_all_handles() or multiple
        calls to (init/grab)_handle should be called before using this function.

        Args:
            serial_num (int): Serial # of the Nano-Drive to search for.

        Returns:
            Returns a valid handle (int).

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_GetHandleBySerial(c_ushort(serial_num))
        if err == 0:
            raise MCL_ND_Exceptions(-8)
        return err

    def release_handle(self, handle):   
        """Releases control of the specified Nano-Drive.

        Args: 
            handle (int): Specifies which Nano-Drive to communicate with.
        """
        self.dll.MCL_ReleaseHandle(handle)

    def release_all_handles(self):
        """Releases control of all Nano-Drives controlled by this instance
        of the DLL.
        """
        self.dll.MCL_ReleaseAllHandles()

    # Standard Device Movement

    def single_read_n(self, axis, handle):
        """
        Read the current position of the specified axis.

        Requirements:
            Firmware version > 0.


        Args:
            axis (int): Which axis to move. (X=1,Y=2,Z=3,AUX=4)
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns a position value (double).

        Raises:
            Raises MCL Exception
        """
        err = self.dll.MCL_SingleReadN(c_uint(axis), c_uint(handle))
        if err in self.error_list:
            raise MCL_ND_Exceptions(err)
        return err

    def single_write_n(self, position, axis, handle):
        """
        Commands the Nano-Drive to move the specified axis to a position.

        Requirements:
            Firmware version > 0.


        Args:
            position (double): Commanded position in microns.
            axis (int): Which axis to move. (X=1,Y=2,Z=3,AUX=4)
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            Raises MCL Exception
        """
        err = self.dll.MCL_SingleWriteN(c_double(position), c_uint(axis), 
                                        handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def monitor_n(self, position, axis, handle):
        """Commands the Nano-Drive to move the specified axis to a position and
        then reads the current position of the axis.

        Requirements:
            Firmware version > 0.


        Args:
            position (double): Commanded position in microns.
            axis (int): Which axis to move.  (X=1,Y=2,Z=3,AUX=4)
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns a position value (double).

        Raises:
            Raises MCL Exception
        """
        err = self.dll.MCL_MonitorN(c_double(position), c_uint(axis), handle)
        if err in self.error_list:
            raise MCL_ND_Exceptions(err)
        return err

    def single_read_z(self, handle):
        """Reads the current position of the Z axis

        Requirements:
            Firmware version > 0.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns a position value (double).

        Raises:
            Raises MCL Exception
        """
        err = self.dll.MCL_SingleReadZ(handle)
        if err in self.error_list:
            raise MCL_ND_Exceptions(err)
        return err

    def single_write_z(self, position, handle):
        """Commands the Nano-Drive to move the Z axis to a position.

        Requirements:
            Firmware version > 0.

        Args:
            position (double): Commanded position in microns.
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            Raises MCL Exception
        """
        err = self.dll.MCL_SingleWriteZ(c_double(position), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def monitor_z(self, position, handle):
        """Commands the Nano-Drive to move the Z axis to a position and 
        then reads the current position of the axis.

        Requirements:
            Firmware version > 0.

        Args:
            position (double): Commanded position in microns.
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns a position value (double).

        Raises:
            Raises MCL Exception
        """
        err = self.dll.MCL_MonitorZ(c_double(position), handle)
        if err in self.error_list:
            raise MCL_ND_Exceptions(err)
        return err

    # Waveform Acquisition

    def read_waveform_n(self, axis, datapoints, milliseconds, handle):
        """This function sets up and triggers a waveform on the specified axis.

        The ADC frequency reverts to the default ADC frequency after calling 
        this function. During a waveform read only the specified axis records 
        data.
        
        (16 bit only) A normal read on the specified axis directly following a
        waveform read will have stale data.  The data will be stale for a few
        milliseconds  (1 ms on single axis systems, 3ms on three axis systems).

        Requirements:
            Firmware version > 0.
            Firmware profile bit 4 equal to 1.  (0x0010)

        Args:
            axis (int): Which axis to move. (X=1,Y=2,Z=3,AUX=4)
            datapoints (int): Number of data points to read.  
                16 bit: Range: 1 - 10000   
                20 bit: Range: 1 - 6666
            milliseconds (double): Rate at which to read data:
                16 bit: Range = 5ms - 1/30ms        
                20 bit: The ADC frequency argument for 20 bit systems is an 
                        index into a table of acceptable values. The following
                        are the valid indexes with their associated time 
                        periods. (index  -> time period)
                        3 -> 267us  , 4 -> 500us  , 5 -> 1ms , 6 -> 2ms
                        7 -> 10ms   , 8 -> 17ms   , 9 -> 20ms
            handle (int): Specifies which Nano-Drive to communicate with.
        
        Returns: 
            Returns a waveform list filled out by this function (doubles list).

        Raises:
            MCL Exception"""
        waveform_list = (c_double * datapoints)()
        err = self.dll.MCL_ReadWaveFormN(c_uint(axis), c_uint(datapoints),
                                         c_double(milliseconds), 
                                         pointer(waveform_list), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return waveform_list

    def setup_read_waveform_n(self, axis, datapoints, milliseconds, handle):
        """Using this functions makes a waveform read a two step process.  
        Setup then Trigger. 

        The Nano-Drive's ADC frequency will not change until the waveform is
        triggered. Performing a setup operation on an already setup axis will 
        overwrite the previous setup.

        Requirements:
            Firmware version > 0.
            Firmware profile bit 4 equal to 1.  (0x0010)

        Args: 
            axis (int): Which axis to move. (X=1,Y=2,Z=3,AUX=4)
            datapoints (int): Number of data points to read.
                16 bit: Range: 1 - 10000   
                20 bit: Range: 1 - 6666
            milliseconds (double): Rate at which to read data:
                16 bit: Range = 5ms - 1/30ms        
                20 bit: The ADC frequency argument for 20 bit systems is an 
                        index into a table of acceptable values. The following
                        are the valid indexes with their associated time 
                        periods. (index  -> time period)
                        3 -> 267us  , 4 -> 500us  , 5 -> 1ms , 6 -> 2ms
                        7 -> 10ms   , 8 -> 17ms   , 9 -> 20ms
            handle (int): Specifies which Nano-Drive to communicate with.
        
        Raises:
            MCL Exception
        """
        err = self.dll.MCL_Setup_ReadWaveFormN(c_uint(axis), c_uint(datapoints), 
                                               c_double(milliseconds), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def trigger_read_waveform_n(self, axis, datapoints, handle):
        """This function triggers a waveform read (previously setup) on the 
        specified axis.

        The axis must have been setup prior to calling this function and 
        datapoints and axis much match their setup values.
        The Nano-Drive's ADC frequency changes prior to data acquisition and 
        reverts to the Nano-Drive's default ADC frequency after calling this 
        function.
        
        During a waveform read only the specified axis records data.
        (16 bit only) A normal read on the specified axis directly following a
        waveform read will have stale data. The data will be stale for a few
        milliseconds (1 ms on single axis systems, 3ms on three axis systems).

        Requirements:
            Firmware version > 0.
            Firmware profile bit 4 equal to 1.  (0x0010)

        Args: 
            axis (int): Which axis to move. (X=1,Y=2,Z=3,AUX=4)
            datapoints (int): Number of data points to read.
                16 bit: Range: 1 - 10000   
                20 bit: Range: 1 - 6666
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns: 
            Returns a waveform list filled out by this function (doubles list).

        Raises:
            MCL Exception
        """
        waveform_list = (c_double * datapoints)()
        err = self.dll.MCL_Trigger_ReadWaveFormN(c_uint(axis), 
                                                 c_uint(datapoints), 
                                                 pointer(waveform_list), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return waveform_list

    def load_waveform_n(self, axis, datapoints, milliseconds, waveform, handle):
        """This function sets up and triggers a waveform load on the
        specified axis.
        
        The Nano-Driver's DAC frequency reverts to the Nano-Drive's default 
        DAC frequency after calling this function.  Only the specified axis will
        have control positions written to it while there is waveform data
        to process.

        Requirements:
            Firmware version > 0.
            Firmware profile bit 4 equal to 1.  (0x0010)

        Args: 
            axis (int): Which axis to move. (X=1,Y=2,Z=3,AUX=4)
            datapoints (int): Number of data points to read.
                                16 bit: Range: 1 - 10000   
                                20 bit: Range: 1 - 6666
            milliseconds (double): Rate at which to read data:
                                16 bit:     Range: 5ms - 1/30ms
                                20 bit: Range: 5ms - 1/6ms
            waveform (doubles list): Array of commanded positions.
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """

        #turn waveform commands into C style array
        array = (c_double * len(waveform))(*waveform)
        err = self.dll.MCL_LoadWaveFormN(c_uint(axis), c_uint(datapoints),
                                         c_double(milliseconds), array, handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def setup_load_waveform_n(self, axis, datapoints, milliseconds, waveform, 
                              handle):
        """This function sets up a waveform load on the specified axis
    
        Nano-Drive's DAC frequency does not change until the waveform is 
        triggered. Performing a setup operation on an already setup axis will 
        overwrite the previous setup

        Requirements:
            Firmware version > 0.
            Firmware profile bit 4 equal to 1. (0x0010)

        Args:
            axis (int): Which axis to move. (X=1,Y=2,Z=3,AUX=4)
            datapoints (int): Number of data points to read.  
                                16 bit: Range: 1 - 10000   
                                20 bit: Range: 1 - 6666
            milliseconds (double): Rate at which to write data.
                                16 bit:     Range: 5ms - 1/30ms
                                20 bit: Range: 5ms - 1/6ms
            waveform (doubles list): Array of commanded positions.
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception

        """
        #turn waveform commands into C style array
        array = (c_double * len(waveform))(*waveform)
        err = self.dll.MCL_Setup_LoadWaveFormN(c_uint(axis), c_uint(datapoints),
                                               c_double(milliseconds),
                                               array, handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def trigger_load_waveform_n(self, axis, handle):
        """This function triggers a waveform load on the specified axis 

        The axis must have been setup prior to calling this function.   
        The Nano-Drive's DAC frequency changes prior to data acquisition and 
        reverts to the Nano-Drive's default DAC frequency after calling this
        function. Only the specified axis will have control positions written to
        it while there is waveform data to process.

        Requirements:
            Firmware version > 0.
            Firmware profile bit 4 equal to 1. (0x0010)

        Args:
            axis (int): Which axis to move.  (X=1,Y=2,Z=3,AUX=4)
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_Trigger_LoadWaveFormN(c_uint(axis), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def trigger_waveform_acquisition(self, axis, datapoints, handle):
        """Triggers a waveform read and a waveform load on the 
        specified axis synchronously

        The axis must have a read & a load waveform setup. 'Datapoints' and 
        'axis' must match the values used to setup the read waveform.
        Only the specified axis will have control positions written to it while
        there is waveform data to process. During a waveform read only the 
        specified axis records data.
        (16 bit only) A normal read on the specified axis directly following a 
        waveform read will have stale data. The data will be stale for a few
        milliseconds (1 ms on single axis systems, 3ms on three axis systems).

        Requirements:
            Firmware version > 0.
            Firmware profile bit 4 equal to 1. (0x0010)

        Args:
            axis (int): Which axis to move.  (X=1,Y=2,Z=3,AUX=4)
            datapoints (int): Number of data points to read.  
                                    16 bit: Range: 1 - 10000   
                                    20 bit: Range: 1 - 6666
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns a waveform list filled out by this function (doubles list).

        Raises:
            MCL Exception
        """
        waveform_list = (c_double * datapoints)()
        err = self.dll.MCL_TriggerWaveformAcquisition(c_uint(axis), 
                                                      c_uint(datapoints), 
                                                      pointer(waveform_list),
                                                      handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return waveform_list

    # Multi Axis Waveform Acquisition

    def wfma_setup(self, wf_dac_x, wf_dac_y, wf_dac_z, data_points_per_axis, 
                   milliseconds, iterations, handle):
        """Prepares the device to run a multi-axis waveform.

        Requirements:
            Firmware profile bit 7 equal to 1. (0x0040)

        Args:
            wf_dac_x (doubles list): Specifies the X axis waveform. 
                Set to NULL if not using the X axis.
            wf_dac_y (doubles list): Specifies the Y axis waveform. 
                Set to NULL if not using the Y axis.
            wf_dac_z (doubles list): Specifies the Z axis waveform. 
                Set to NULL if not using the Z axis.
            data_points_per_axis (int): Number of data points in each axis'
                waveform.  The total number of data points for all axes must be 
                below 10,000 for 16-bit systems or below 6,666 on 20-bit 
                systems. For example, the maximum number of data points per axis
                when using all three axes on a 20-bit system is 2,222.
            milliseconds (double): Specifies the amount of time between
                data points.
                16 bit: Range: 5ms - 1/10ms
                20 bit: The milliseconds argument for 20 bit systems is an index
                    into a table of acceptable values.  The following are the 
                    valid indexes with their associated time periods.
                    3.0 -> 267us, 4.0 -> 500us , 5.0 -> 1ms ,  6.0 -> 2ms
            iterations (int): The number of times to run the waveform.  
                Entering 0 specifies infinite iterations.
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        if wf_dac_x is None:
            x_waveform = None
        else:
            x_waveform = (c_double * data_points_per_axis)(*wf_dac_x)
            
        if wf_dac_y is None:
            y_waveform = None
        else:
            y_waveform = (c_double * data_points_per_axis)(*wf_dac_y)
            
        if wf_dac_z is None:
            z_waveform = None
        else:
            z_waveform = (c_double * data_points_per_axis)(*wf_dac_z)
        self.__wfma_len = data_points_per_axis
        err = self.dll.MCL_WfmaSetup(x_waveform, y_waveform, z_waveform, 
                                     data_points_per_axis, 
                                     c_double(milliseconds),
                                     c_ushort(iterations), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def wfma_trigger_and_read(self, handle):
        """ Triggers the multi-axis waveform and waits for it to finish.

        Only the last iteration of ADC data is saved.
        If the number of waveform iterations is 0 (INFINITE) an error will be
        returned. Use wfma_trigger and wfma_read to handle infinite waveforms.

        Requirements:
            Firmware profile bit 7 equal to 1. (0x0040)

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            X ADC data from the multi-axis waveform (doubles list).
            Y ADC data from the multi-axis waveform (doubles list).
            Z ADC data from the multi-axis waveform (doubles list).

        Raises:
            MCL Exception
        """
        wf_adc_x = (c_double * self.__wfma_len)()
        wf_adc_y = (c_double * self.__wfma_len)()
        wf_adc_z = (c_double * self.__wfma_len)()

        err = self.dll.MCL_WfmaTriggerAndRead(pointer(wf_adc_x), 
                                              pointer(wf_adc_y),
                                              pointer(wf_adc_z), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return wf_adc_x, wf_adc_y, wf_adc_z

    def wfma_trigger(self, handle):
        """Starts the multi-axis waveform.

        Requirements:
            Firmware profile bit 7 equal to 1. (0x0040)

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_WfmaTrigger(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def wfma_read(self, handle):
        """Reads the results of a multi-axis waveform.  If the waveform is still
        running it will wait until it is complete or stop it if the waveform was
        setup to run continuously.
        
        Only the last iteration of ADC data is saved.

        Requirements:
            Firmware profile bit 7 equal to 1. (0x0040)

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns: 
            X ADC data from the multi-axis waveform (doubles list).
            Y ADC data from the multi-axis waveform (doubles list).
            Z ADC data from the multi-axis waveform (doubles list).

        Raises:
            MCL Exception
        """
        wf_adc_x = (c_double * self.__wfma_len)()
        wf_adc_y = (c_double * self.__wfma_len)()
        wf_adc_z = (c_double * self.__wfma_len)()

        err = self.dll.MCL_WfmaRead(pointer(wf_adc_x), pointer(wf_adc_y),
                                    pointer(wf_adc_z), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return wf_adc_x, wf_adc_y, wf_adc_z

    def wfma_stop(self, handle):
        """Stops a multi-axis waveform.

        Requirements:
            Firmware profile bit 7 equal to 1. (0x0040)

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_WfmaStop(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    # ISS Option


    
    def pixel_clock(self, handle):
        """Generates a 250 ns pulse on the pixel clock

        This function allows programmatic control of the pixel clock, one of the
        four external clocks that makes up the ISS option. The polarity of these
        pulses can be configured with iss_configure_polarity. By default, all
        pulses will be low to high. 

        Using iss_set_clock to set a clock high or low will have an effect on
        this function.  For example, if a clock is set high and has its default
        polarity the "pulse" will be seen as a falling edge and the clock will
        remain low after this function.

        Requirements:
            Firmware version > 0.
            Image Scan Sync (ISS) option.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_PixelClock(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def line_clock(self, handle):
        """Generates a 250 ns pulse on the line clock

        This function allows programmatic control of the line clock, one of the 
        four external clocks that makes up the ISS option. The polarity of these
        pulses can be configured with iss_configure_polarity. By default, all
        pulses will be low to high. 

        Using iss_set_clock to set a clock high or low will have an effect on 
        this function.  For example, if a clock is set high and has its default 
        polarity the "pulse" will be seen as a falling edge and the clock will 
        remain low after this function.

        Requirements:
            Firmware version > 0.
            Image Scan Sync (ISS) option.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_LineClock(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def frame_clock(self, handle):
        """Generates a 250 ns pulse on the frame clock

        This function allows programmatic control of the pixel clock, one of the
        four external clocks that makes up the ISS option. The polarity of these
        pulses can be configured with iss_configure_polarity. By default, all
        pulses will be low to high. 

        Using iss_set_clock to set a clock high or low will have an effect on 
        this function.  For example, if a clock is set high and has its default 
        polarity the "pulse" will be seen as a falling edge and the clock will
        remain low after this function.

        Requirements:
            Firmware version > 0.
            Image Scan Sync (ISS) option.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_FrameClock(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def aux_clock(self, handle):
        """Generates a 250 ns pulse on the frame clock

        This function allows programmatic control of the pixel clock, one of the
        four external clocks that makes up the ISS option. The polarity of these
        pulses can be configured with iss_configure_polarity. By default, all
        pulses will be low to high. 

        Using iss_set_clock to set a clock high or low will have an effect on 
        this function.  For example, if a clock is set high and has its default
        polarity the "pulse" will be seen as a falling edge and the clock will 
        remain low after this function.

        Requirements:
            Firmware version > 0.
            Image Scan Sync (ISS) option.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_AuxClock(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def iss_set_clock(self, clock, mode, handle):
        """Sets an external clock high or low. 

        Using MCL_IssSetClock will unbind the specified clock from all axes; the
        specified clock will not be unbound from waveform read/write events.
        If you have the Pixel clock bound to the X axis, Y axis,
        and waveform read and then use MCL_IssSetClock to set the 
        pixel clock high, the Pixel clock will no longer be bound to 
        the X axis or Y axis but will remain bound to the waveform read event.

        Requirements:
            Firmware version > 0.
            Firmware profile bit 0 equal to 1. (0x0001)
            Image Scan Sync (ISS) option.

        Args:
            clock (int): Which clock to set.  (1=Pixel, 2=Line, 3=Frame, 4=Aux)
            mode  (int): Determines whether to set the clock high or low. 
                            0 = Sets clock low.
                            1 = Sets clock high.
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_IssSetClock(clock, mode, handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def iss_reset_defaults(self, handle):
        """Resets the Iss option to its default values. 
       
        No axis is bound.  All polarities are low to high. The Pixel clock is 
        bound to the waveform read event. The line clock is bound to 
        the waveform write event.

        Requirements:
            Firmware version > 0.
            Firmware profile bit 0 equal to 1. (0x0001)
            Image Scan Sync (ISS) option.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception   
        """
        err = self.dll.MCL_IssResetDefaults(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def iss_bind_clock_to_axis(self, clock, mode, axis, handle):
        """Allows an external clock pulse to be bound to the read 
        of a particular axis. 
       
        Clocks may also be bound to portions of the waveform functionality.  
        A clock can be bound to the data acquisition of a read waveform so that 
        every time a point is recorded a clock pulse is generated. Additionally,
        a clock can be pulsed prior to the first point and pulsed after the last
        point of a position command waveform.

        Each axis read and waveform event can only be bound to one external 
        clock.  Attempting to bind a second clock to an axis will simply replace
        the existing bind with the newer bind.

        Requirements:
            Firmware version > 0.
            Firmware profile bit 0 equal to 1. (0x0001)
            Image Scan Sync (ISS) option.

        Args:
            clock (int): Which clock to bind.  (1=Pixel, 2=Line, 3=Frame, 4=Aux)
            mode  (int): Selects polarity of the clock to be bound 
                or selects to unbind the axis. 
                                2 = low to high pulse.
                                3 = high to low pulse.
                                4 = unbind the axis.
            axis (int): Axis or event to bind a clock to.
                                1  = X axis
                                2  = Y axis
                                3  = Z axis
                                4  = Aux axis
                                5 = Waveform Read.
                                6 = Waveform Write.
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception 
        """
        err = self.dll.MCL_IssBindClockToAxis(clock, mode, axis, handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)


    def iss_configure_polarity(self, clock, mode, handle):
        """Configures the polarity of the external clock pulses 
        generated by (pixel/line/frame/aux)_clock().

        Does not affect the polarity of clocks bound to axes or events by 
        iss_bind_clock_to_axis. If you set the polarity of the Pixel clock to 
        be high to low using iss_configure_polarity and bind the Pixel clock to
        the X axis as a low to high pulse then calling pixel_clock will generate
        a high to low pulse and reading the X axis will generate 
        a low to high pulse.

        Requirements:
            Firmware version > 0.
            Firmware profile bit 0 equal to 1. (0x0001)
            Image Scan Sync (ISS) option.

        Args:
            clock (int): Which clock to configure. 
                (1=Pixel, 2=Line, 3=Frame, 4=Aux)
            mode  (int): Selects polarity of the clock. 
                2 = low to high pulse.
                3 = high to low pulse.
            handle: Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_IssConfigurePolarity(clock, mode, handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    # C-Focus

    def c_focus_set_focus_mode(self, focus_mode_on, handle):
        """This function replicates the functionality of the 
        front panel 'focus lock' button.

        If the C-Focus system is already changing its operating mode the 
        function will return MCL_DEV_NOT_READY. It takes about 5 seconds 
        to change the operating mode.

        Requirements:
            This function is specific to product id 0x2401.
            Firmware profile bit 2 equal to 1. (0x0004)

        Args:
            focus_mode_on (bool): true - puts the C-Focus into focus lock mode.
                                  false - puts the C-Focus into normal mode.
            handle (int): Specifies which device to communicate with.

        Raises:
            MCL Exception

        """
        err = self.dll.MCL_CFocusSetFocusMode(c_bool(focus_mode_on), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def c_focus_step(self, relative_position_change, handle):
        """Steps the nanopositioner while in focus mode and refocuses at 
        the new position.

        The resolution of the relative position change is limited to the 
        resolution of C-Focus system's encoder.

        Requirements:
            This function is specific to product id 0x2401.
            Firmware profile bit 2 equal to 1. (0x0004)

        Args:
            relative_position_change (double): Distance to move the 
                nanopositioner from its current position.
            handle (int): Specifies which device to communicate with.

        Raises: 
            MCL Exception
        """
        err = self.dll.MCL_CFocusStep(c_double(relative_position_change), 
                                      handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def c_focus_get_focus_mode(self, handle):
        """Queries the state of the C-Focus.

        Requirements:
            This function is specific to product id 0x2401.
            Firmware profile bit 2 equal to 1. (0x0004)

        Args:
            handle (int): Specifies which device to communicate with.

        Returns:
            1 if focus is locked or 0 otherwise.

        Raises:
            MCL Exception
        """
        focus_locked = c_int32()
        err = self.dll.MCL_CFocusGetFocusMode(byref(focus_locked), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return focus_locked


    # Encoder

    def read_encoder_z(self, handle):
        """Reads the current value of the Z encoder.

        Requirements:
            This function is specific to product ids 0x2000, 0x2100, 0x2401.

        Args:
            handle (int): Specifies which device to communicate with.

        Returns:
            Returns the current value of the Z encoder (double).

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_ReadEncoderZ(handle)
        if err in self.error_list:
            raise MCL_ND_Exceptions(err)
        return err

    def reset_encoder_z(self, handle):
        """Resets the Z encoder to 0.

        Requirements:
            This function is specific to product ids 0x2000, 0x2100, 0x2401.

        Args:
            handle (int): Specifies which device to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_ResetEncoderZ(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    # Tip Tilt Z

    def theta_x(self, milliradians, handle):
        """Calculates the required positioning of A, B, and C to form 
        a theta-X of 'milliradians' 
        
        Calculates the required positioning of A, B, and C to form a theta-X of
        'milliradians' based on the dimensions of the actuator triangle.    
        If achieving a theta-X of 'milliradians' is not possible given the 
        current position of the actuators, the function will coerce 
        'milliradians' to the maximum possible (or minimum if 'milliradians'
        is negative) currently achievable theta-X. 

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            milliradians (double): Desired theta-X in milliradians.
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns the angle in milliradians produced by this function (double)

        Raises:
            MCL Exception
        """
        actual = c_double()
        err = self.dll.MCL_ThetaX(c_double(milliradians), byref(actual), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return actual.value

    def theta_y(self, milliradians, handle):
        """Calculates the required positioning of A and B to form 
        a theta-Y of 'milliradians' based on the dimensions of the
        actuator triangle. If achieving a theta-Y of 'milliradians' is not 
        possible given the current position of the actuators the function will 
        coerce 'milliradians' to the maximum possible (or minimum if 
        'milliradians' is negative) currently achievable theta-Y.

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            milliradians (double): Desired theta-Y in milliradians.
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns the angle in milliradians produced by this function (double)

        Raises:
            MCL Exception
        """
        actual = c_double()
        err = self.dll.MCL_ThetaY(c_double(milliradians), byref(actual), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return actual.value

    def move_z_center(self, position, handle):
        """Moves the center point (the midpoint of the perpendicular line 
        segment from C to line segment AB) to a height of 'position' maintaining
        the current theta-X and theta-Y of the plane. If 'position' is not 
        achievable without changing theta-X and/or theta-Y the function will 
        move the center point to the maximum (or minimum) possible position such
        that theta-X and theta-Y remain unchanged.

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            position (double): Desired center point in microns.
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns achieved height of the center point in microns (double).

        Raises:
            MCL_Exception
        """
        actual = c_double()
        err = self.dll.MoveZCenter(c_double(position), byref(actual), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return actual.value

    def level_z(self, position, handle):
        """Moves all the actuators to 'position' achieving a level plane.

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            position (double): Z height to move to.
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_LevelZ(c_double(position), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def tip_tilt_height(self, handle):
        """Returns the height of the actuator triangle in millimeters.

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns height of the actuator triangle (double).

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_TipTiltHeight(handle)
        if err in self.error_list:
            raise MCL_ND_Exceptions(err)
        return err

    def tip_tilt_width(self, handle): 
        """Returns the width of the actuator triangle in millimeters.

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns width of the actuator triangle (double). 

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_TipTiltWidth(handle)
        if err in self.error_list:
            raise MCL_ND_Exceptions(err)
        return err

    def min_max_theta_x(self, handle):
        """Calculates the maximum and minimum achievable theta-X angles 
        in milliradians.

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Minimum achievable theta-X in milliradians (double).
            Maximum achievable theta-X in milliradians (double).

        Raises:
            MCL Exception
        """
        minimum = c_double()
        maximum = c_double()
        err = self.dll.MCL_MinMaxThetaX(byref(minimum), byref(maximum), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return minimum.value, maximum.value

    def min_max_theta_y(self, handle):
        """Calculates the maximum and minimum achievable theta-Y angles
        in milliradians.

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns minimum achievable theta-Y in milliradians (double).
            Returns maximum achievable theta-Y in milliradians (double).

        Raises:
            MCL Exception
        """
        minimum = c_double()
        maximum = c_double()
        err = self.dll.MCL_MinMaxThetaY(byref(minimum), byref(maximum), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return minimum.value, maximum.value

    def get_tip_tilt_theta_x(self, handle):
        """Returns the current theta-X of the plane.

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns the current theta-X of the plane (double).

        Raises:
            MCL Exception
        """         
        err = self.dll.MCL_GetTipTiltThetaX(handle)
        if err in self.error_list:
            raise MCL_ND_Exceptions(err)
        return err

    def get_tip_tilt_theta_y(self, handle):
        """Returns the current theta-X of the plane.

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns the current theta-X of the plane (double).

        Raises:
            MCL Exception
        """ 
        err = self.dll.MCL_GetTipTiltThetaY(handle)
        if err in self.error_list:
            raise MCL_ND_Exceptions(err)
        return err

    def get_tip_tilt_center(self, handle):
        """Returns the current height of the center point (the midpoint of the 
        perpendicular line segment from C to line segment AB).

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns the height of the center point (double).

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_GetTipTiltCenter(handle)
        if err in self.error_list:
            raise MCL_ND_Exceptions(err)
        return err
    
    def current_min_max_theta_x(self, handle):
        """Given the position of A, B, and C calculates the minimum and 
        maximum achievable theta-X.

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns minimum achievable theta-X in milliradians (double).
            Returns maximum achievable theta-X in milliradians (double).

        Raises: 
            MCL Exception
        """
        minimum = c_double()
        maximum = c_double()
        err = self.dll.MCL_CurrentMinMaxThetaX(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return minimum.value, maximum.value

    def current_min_max_theta_y(self, handle):
        """Given the position of A, B, and C calculates the minimum and
        maximum achievable theta-Y.

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns minimum achievable theta-Y in milliradians (double).
            Returns maximum achievable theta-Y in milliradians (double).

        Raises: 
            MCL Exception
        """
        minimum = c_double()
        maximum = c_double()
        err = self.dll.MCL_CurrentMinMaxThetaY(handle)
        if err != 0:
            raise MCL_ND_Exceptions
        return minimum.value, maximum.value

    def current_min_max_center(self, handle):
        """Given the position of A, B, and C calculates the minimum
        and maximum achievable center point (the midpoint of the perpendicular 
        line segment from C to line segment AB).

        Requirements:
            This function is specific to product ids 0x1253 and 0x2253.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns minimum achievable center height in microns (double).
            Returns maximum achievable center height in microns (double).

        Raises:
            MCL Exception
        """
        minimum = c_double()
        maximum = c_double()
        err = self.dll.MCL_CurrentMinMaxThetaY(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return minimum.value, maximum.value

    # Sequences

    def sequence_load(self, axis, sequence, seq_size, handle):
        """Writes a sequence of positions to the Nano-Drive. This sequence may 
        later be stepped through by sending the Nano-Drive a TTL pulse.

        Requirements:
            Firmware version > 0.
            Firmware profile bit 8 equal to 1. (0x0100)

        Args:
            axis (int): Which axis to move. (X=1,Y=2,Z=3,AUX=4)
            sequence (doubles list): List of positions to send to the Nano-Drive.
            seq_size (int): Number of positions in sequence.
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        array = (c_double * seq_size)(*sequence)
        err = self.dll.MCL_SequenceLoad(axis, array, seq_size, handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def sequence_clear(self, handle):
        """Stops and clears the sequence on the Nano-Drive controller. 
        
        The old sequence is completely discarded. A new sequence must be loaded
        prior to starting again.

        Requirements:
            Firmware version > 0.
            Firmware profile bit 8 equal to 1. (0x0100)

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_SequenceClear(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        

    def sequence_start(self, handle):
        """Once a sequence has been loaded, starting it will allow the 
        Nano-Drive to respond to external interrupts and execute
        the loaded sequence.

        Requirements:
            Firmware version > 0.
            Firmware profile bit 8 equal to 1. (0x0100)

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCLSequenceStart(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def sequence_stop(self, handle):
        """Stops the sequence on the Nano-Drive controller.
       
        The Nano-Drive will not respond to external TTL pulses when stopped.
        Stopping will reset the index in the sequence to the beginning
        of the sequence.

        Requirements:
            Firmware version > 0.
            Firmware profile bit 8 equal to 1. (0x0100)

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCLSequenceStop(handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def sequence_get_max(self, handle):
        """Queries the Nano-Drive to determine how many sequence positions
        are supported.

        Requirements:
            Firmware version > 0.
            Firmware profile bit 8 equal to 1. (0x0100)

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns max number of sequence positions (int).

        Raises:
            MCL Exception
        """
        maximum = c_int32()
        err = self.dll.MCL_SequenceGetMax(byref(maximum), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return maximum.value

    # Clock Functionality

    def change_clock(self, milliseconds, clock, handle):
        """Changes the frequency the Nano-Drive reads or writes data.

        Requirements:
            Firmware version > 0.

        Args:
            milliseconds (double): 
                ADC frequency (16 bit)
                    Single axis: Range: 5ms - 1/30ms
                    Multi-axis: Range: 5ms - 1/10ms
                ADC frequency (20 bit)
                    The ADC frequency argument for 20 bit systems is an index 
                    into a table of acceptableceptable values.  The following 
                    are the valid indexes with their associated time periods.
                        3 -> 267us  , 4 -> 500us  , 5 -> 1ms , 6 -> 2ms
                        7 -> 10ms   , 8 -> 17ms   , 9 -> 20ms
                DAC frequency  (16 bit)
                    Single axis:    Range: 5ms - 1/30ms
                    Multi-axis:     Range: 5ms - 1/10ms
                DAC frequency (20 bit)
                    All:            Range: 5ms - 1/6ms
            clock (int): 0 - ADC(reading data)
                   1 - DAC(writing data)
            handle (int): Specifies which Nano-Drive to communicate with.

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_ChangeClock(c_double(milliseconds), c_short(clock),
                                       handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)

    def get_clock_frequency(self, handle):
        """Allows user to see the Nano-Drive's clock frequencies in milliseconds 

        Requirements:
            Firmware version > 0.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns Nano-Drive's ADC frequency in milliseconds (double).
            Returns Nano-Drive's DAC frequency in milliseconds (double).
        """
        adc_freq = c_double()
        dac_freq = c_double()
        err = self.dll.MCL_GetClockFrequency(byref(adc_freq), byref(dac_freq), 
                                             handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        else:
            return adc_freq.value, dac_freq.value

    # Device

    def device_attached(self, milliseconds, handle):
        """Function waits for a specified number of milliseconds then reports
        whether or not the Nano-Drive is attached. If a device has been detached
        the current handle should be released.

        Args:
            milliseconds (int): Indicates how long to wait.
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns true if the specified Nano-Drive is attached and 
            false if it is not.
        """
        return self.dll.MCL_DeviceAttached(c_uint(milliseconds), handle)

    def get_calibration(self, axis, handle):
        """Returns the range of motion of the specified axis.

        Requirements:
            Firmware version > 0.

        Args:
            axis (int): Which axis to get the calibration of.(X=1,Y=2,Z=3,AUX=4)
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns the range of motion (double).
        """
        err = self.dll.MCL_GetCalibration(c_uint(axis), handle)
        if err in self.error_list:
            raise MCL_ND_Exceptions(err)
        return err

    def get_firmware_version(self, handle):
        """Gives access to the Firmware version and profile information.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns firmware version number (int).
            Returns firmware profile number (int).

        Raises:
            MCL Exception
        """
        version = c_short()
        profile = c_short()
        err = self.dll.MCL_GetFirmwareVersion(byref(version), byref(profile),
                                              handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return version.value, profile.value

    def print_device_info(self, handle):
        """Prints to the console product name, product id, DLL version, 
        firmware version, and some product specific information.
        """
        return self.dll.MCL_PrintDeviceInfo(handle)

    def get_serial_number(self, handle):
        """Returns the serial number of the Nano-Drive.  This information can be
        useful if you need support for your device or if you are attempting to 
        tell the difference between two similar Nano-Drives.

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns the serial number of the specified Nano-Drive (int).

        Raises:
            MCL Exception
        """
        err = self.dll.MCL_GetSerialNumber(handle)
        if err in self.error_list:
            raise MCL_ND_Exceptions
        return err

    def dll_version(self):
        """Gives access to the DLL version information.  This information is 
        useful if you need support.

        Returns:
            Returns DLL version number  (int).
            Returns DLL revision number (int).
        """
        version = c_short()
        revision = c_short()
        self.dll.MCL_DLLVersion(byref(version), byref(revision))
        return version.value, revision.value

    def correct_driver_version(self):
        """Checks if the DLL was built against the driver version currently
        installed.

        Returns:
            True if the DLL was built against the current driver version.
        """
        return self.dll.MCL_CorrectDriverVersion()

    def get_commanded_position(self, handle):
        """Finds the current commanded position for the X, Y, and Z axis.

        Requirements:
            Firmware profile bit 3 equal to 1. (0x0008)

        Args:
            handle (int): Specifies which Nano-Drive to communicate with.

        Returns:
            Returns commanded position of the X axis (double).
            Returns commanded position of the Y axis (double).
            Returns commanded position of the Z axis (double).

        Raises:
            MCL Exception
        """
        x_com = c_double
        y_com = c_double
        z_com = c_double
        err = self.dll.MCL_GetCommandedPosition(byref(x_com), byref(y_com), 
                                                byref(z_com), handle)
        if err != 0:
            raise MCL_ND_Exceptions(err)
        return x_com.value, y_com.value, z_com.value

    def exception_test(self, num):
        if num != 0:
            raise MCL_ND_Exceptions(num)
        return num

class MCL_ND_Exceptions(Exception):
    def __init__(self, err):
        error_messages = {
            -1 : 'MCL General Error occurred: -1',
            -2 : 'MCL Device Error occurred: -2',
            -3 : 'MCL Device Not Attached: -3',
            -4 : 'MCL General Error occurred: -4',
            -5 : 'MCL Usage Error occurred: -5',
            -6 : 'MCL Argument Error occurred: -6',
            -7 : 'MCL Invalid Axis: -7',
            -8 : 'MCL Invalid Handle: -8'
        }
        message = error_messages.get(err, f"Unknown error with code {err}")
        super().__init__(message)

