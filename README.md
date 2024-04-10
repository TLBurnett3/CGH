# CGH
Demonstration of CGH interference pattern/intensity image rendering from a point cloud.  Note: The produced wavefield has not been verified on an real holographic display.

## Source Code
* Src/Core - Contains the base code for the project
* Src/Common - Contains a common classes
* Src/CGH - Contains the main and executor for this project

## Build using CMake
Building this project is a two-step process.  First, [3rdPartyLib](https://github.com/TLBurnett3/3rdPartyLibs) is required to be built and installed on the development/build platform.  Second, using CMake a platform specific make solution must be built and executed to create a build solution.  Currently, only Windows has been tested.

## Running CGH
The CGH executable requires a JSon file for defining the execution parameters.  An example is included within the Cfg directory.

For Example: CGH ./Cfg/Bunny/Bunny.json

During execution, CGH generates a ProofImage.png and the QStat.png.  The QStat.png is a row record of rendered rows.  If CGH is interrupt, the QStat.png informs CGH where to start rendering again.  It must be deleted to re-render a full dataset.
When the WaveField parameter is on, the pixel wavefront is recorded by row into the "OutPath/WaveField" directory.  See below.

### JSon File Description

##### JobName
Name of the Job.

##### OutPath
Output directory.

#### Dim
Dimensions of the hologram image in millimeters defined on an X,Y plane.

#### PixelSize
Size of a hologram pixel in millimeters.

#### FoV
Field of view of a holographic pixel.

#### NumThreads
Number of threads to use for calculating the intensity image.

#### PointCloud
The pointcloud .pcd file.

#### WaveLengths
The RGB wavelengths for wavefield calculation.

#### WaveField
Generate the RGBW wavefield for each holographic pixel row.

#### PCTransform
A transform matrix applied to the point cloud.


