*     Simple wrappers to open and close files fortran-style

      subroutine openfile( iunit, name, inform )
      implicit none
      integer iunit
      character*(*) name
      integer inform

      open( iunit, iostat=inform, file=name, status='replace' )

      end



      subroutine closefile( iunit )
      implicit none
      integer iunit

      close( iunit )

      end
