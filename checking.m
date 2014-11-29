function c = checking(script_name)

load([script_name '_checker.mat'], 'check_data');
c = 0;
for i=1:size(check_data,2)
    try
        func = check_data{i}.func;
        n_in = size(func.inputs,2);
        n_out = size(func.outputs,2);
        input = check_data{i}.input;
        output_size = check_data{i}.output_size;
        for j=1:n_in
            in_args{j} = input.(func.inputs{j});
        end
        [out_args{1:n_out}] = feval(func.name, in_args{:});
        
        for j=1:n_out            
            if norm(size(out_args{j}) - output_size{j})
                fprintf(1, 'WRONG OUTPUT: [%d x %d], expected [%d x %d]\n', ...
                        size(out_args{j}), output_size{j});
                c = c + 1;
            end
        end
        clear('in_args');
        clear('out_args');
    catch ME
        c = c + 1;
        fprintf(1, 'ERROR. Cannot execute %s\n', check_data{i}.func.name);
        ME
    end
end

if c==0
    fprintf(1,'successfully passed size tests!\n');
end

end