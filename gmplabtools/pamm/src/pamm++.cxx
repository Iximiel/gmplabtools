/*
 Copyright (C) 2022, Daniele Rapetti

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDIng BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRIngEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 TORT OR OTHERWISE, ARISIng FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*    return dict(
        # cluster
        size=gridsize,
        p=2,
        generate_grid=True,
        savegrid=f"grid_{pcaTrainFname}",X
        d=dimension,
        fspread=fspread,  # gaussian sigma sigma
        ngrid=gridsize,
        qs=1,
        o=outFnames,
        trajectory=f"./{pcaTrainFname}.pca.tmp",
        merger=merger,  # used to be 0.05,
        bootstrap=bootstrap,
    )
    Executing command: pamm -d 3 -bootstrap 73 -fspread 0.25 -qs 1 -o pamm
   -ngrid 2000 -merger 0.005 -v < ./.pca.tmp
    */

#include "libpamm++.hpp"

int main(int /*argc*/, char ** /*argv*/) {
  libpamm::clusteringMode();
  return 0;
}