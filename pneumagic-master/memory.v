module memory_1bit(
                   input      clk,
                   input      write_enable,
                   input      address,
                   input      data_in,
                   output reg data_out
                   );

   // Define memory array (2 locations, each 1-bit wide)
   reg [0:1] mem;

   // Read operation (asynchronous)
   always @(*) begin
      data_out = mem[address];
   end

   // Write operation (synchronous)
   always @(posedge clk) begin
      if (write_enable)
        mem[address] <= data_in;
   end

endmodule
