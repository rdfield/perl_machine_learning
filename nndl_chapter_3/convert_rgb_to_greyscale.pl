use Modern::Perl;

my $input_file = shift;
open FILE,"<",$input_file or die $!;
open OUTFILE,">",substr($input_file,0,-4) . ".dat";
my $ctr = 0;
my $header = <FILE>; # skip header line
while (<FILE>) {
  my @elements = split;
  my $red = substr($elements[2],1,2);
  my $green = substr($elements[2],3,2);
  my $blue = substr($elements[2],5,2);
  $red = hex($red);
  $green = hex($green);
  $blue = hex($blue);
  my $grey = 0.299 * $red + 0.587 * $green + 0.114 * $blue;
  if ($grey < 59) {
     print OUTFILE chr(0);
  } else {
     print OUTFILE chr($grey);
  }
}
close FILE;
close OUTFILE;
