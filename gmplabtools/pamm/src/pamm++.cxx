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
#include <iostream>

using std ::cout;
using std::endl;
int main(int /*argc*/, char ** /*argv*/) {
  constexpr size_t DIM = 8;
  libpamm::distanceMatrix t(DIM);
  cout << "Dimension: " << ((DIM - 1) * DIM) / 2 << endl << endl;
  int TT = 0;
  for (size_t i = 0; i < DIM; i++) {
    for (size_t j = 0; j < i; j++) {
      t(i, j) = TT;
      // cout << i << ", " << j << ": " << t.address(i, j) << " -> " << TT << "
      // ("<< int(i * (i - 1) / 2) << ")" << endl;
      ++TT;
    }
  }
  for (size_t i = 0; i < DIM; i++) {
    for (size_t j = 0; j < i; j++) {
      cout << t(j, i) << " ";
    }
    cout << " n ";
    for (size_t j = i + 1; j < DIM; j++) {

      cout << t(j, i) << " ";
    }
    cout << endl;
  }
  return 0;
}