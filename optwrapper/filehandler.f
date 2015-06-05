*     Simple wrappers to open and close files fortran-style

      subroutine openfile( iunit, name, inform )
      implicit none
      integer iunit
      character*(*) name
      integer inform

      open( UNIT=iunit, IOSTAT=inform, FILE=name, STATUS="replace" )

      end



      subroutine closefile( iunit )
      implicit none
      integer iunit

      close( UNIT=iunit )

      end
