use Modern::Perl;

my $input_file = shift;
open FILE,"<",$input_file or die $!;
open OUTFILE,">",substr($input_file,0,-4) . ".dat";
foreach my $i (0 .. (4 * 28 - 1)) {
   print OUTFILE chr(0);
}
my $ctr = 0;
my $header = <FILE>; # skip header line
while (<FILE>) {
  if ($ctr == 0) {
     say "pixel $ctr, pre padding with 0000";
     print OUTFILE chr(0) . chr(0) . chr(0) . chr(0);
  }
  my @elements = split;
  print $elements[2] . " = ";
  my $red = substr($elements[2],1,2);
  my $green = substr($elements[2],3,2);
  my $blue = substr($elements[2],5,2);
  print " red = $red green = $green blue = $blue ";
  $red = hex($red);
  $green = hex($green);
  $blue = hex($blue);
  print " red = $red green = $green blue = $blue ";
  my $grey = 0.299 * $red + 0.587 * $green + 0.114 * $blue;
  print " grey = $grey\n";
  if ($grey < 59) {
     print OUTFILE chr(0);
  } else {
     print OUTFILE chr($grey);
  }
  if ($ctr == 19) {
     say "pixel $ctr, post padding with 0000";
     print OUTFILE chr(0) . chr(0) . chr(0) . chr(0);
     $ctr = 0
  } else {
    $ctr++;
  }
}
foreach my $i (0 .. (4 * 28 - 1)) {
   print OUTFILE chr(0);
}
close FILE;
close OUTFILE;
