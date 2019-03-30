
package yuxizh;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;

/**
 * 模型函数 f(x;a,b) = a * x + b
 */
class ModelFunction {

    /**
     * 函数求值
     * @param x 函数自变量向量
     * @param args 函数参数向量
     * @return 函数因变量浮点值
     */
    public float evaluate(float[] x, float[] args) {
        float result = 0.0f;
        for (int i = 0; i < x.length; ++i) {
            result += args[i] * x[i];
        }
        return result;
    }

    /**
     * 函数对第i维参数的偏导函数求值
     * @param i 对第i维参数求偏导
     * @param x 偏导数函数自变量向量
     * @param args 偏导数函数自变量
     * @return 偏导函数引变量浮点值
     */
    public float partialDiffWith(int i, float[] x, float[] args) {
        return x[i] ;
    }

    // df(x)/da = x
    // public float partialDiffWithA(float x, float[] args) {
        // return x;
    // }

    // df(x)/db = 1
    // public float partialDiffWithB(float x, float[] args) {
        // return 1.0f;
    // }
}

/**
 * 目标函数 J(a,b;X) = \frac{1}{|X|}\sum_{(x,y) \in X}(f(x;w) - y)^2
 */
class TargetFunction {

    private ModelFunction model;

    public TargetFunction(ModelFunction model) {
        this.model = model;
    }

    // dJ(w;X)/dw_i = \frac{1}{|X|}\sum_{(x,y) \in X}2(f(x;w) - y)df(x;w)/dw_i
    public float partialDiffWith(int i, float[] args, float[][] X, float[] Y) {
        float sum = 0.0f;
        for (int j = 0; j < X.length; ++j) {
            sum += 2 * (model.evaluate(X[j], args) - Y[j]) * model.partialDiffWith(i, X[j], args);
        }
        return sum / X.length;
    }
}

abstract class Solver {

    // 目标函数
    protected TargetFunction target;

    // 样本数据集
    protected float[][] dataset;

    // 样本标签集
    protected float[] label;

    // 前次计算所得参数
    protected float[] prevArgs;

    // 当次计算所得参数
    protected float[] currArgs;

    public Solver(TargetFunction target, float[][] dataset, float[] label) {
        this.target = target;
        this.dataset = dataset;
        this.label = label;
        prevArgs = new float[5];
        currArgs = new float[5];
    }

    // 梯度下降求解
    final public void solve(float alpha, float beta, int numIterations) {
        // 迭代次数计数器
        int cntIterations = 0;
        do {
            // 计算档次参数值
            update(alpha);
            // 技术器自增
            cntIterations += 1;
        } while (hasConverged(beta) || cntIterations == numIterations);
    }

    // 连续两次计算所得参数的无穷范数小于beta则认为已经收敛
    final private boolean hasConverged(float beta) {
        for (int i = 0; i < prevArgs.length; ++i) {
            if (Math.abs(prevArgs[i] - currArgs[i]) > beta) {
                return false;
            }
        }
        return true;
    }

    final protected void swapArgs() {
        float[] temp = null;
        temp = prevArgs;
        prevArgs = currArgs;
        currArgs = temp;
    }

    final public float[] getResult() {
        return prevArgs;
    }

    abstract protected void update(float alpha);
}

/**
 * 全量梯度下降求解器
 * \begin{aligned}
 * &\text{do} \\
 * &\quad\theta_{i+1}=\theta_i-\alpha \nabla J(\theta_{i};X)\\
 * &\text{until } ||\theta_{i+1} - \theta_{i}||_{\infty}\le\beta \text{ OR iterate N times}
 * \end{aligned}
 */
final class BGDSlover extends Solver {

    public BGDSlover(TargetFunction target, float[][] dataset, float[] label) {
        super(target, dataset, label);
    }

    @Override
    final protected void update(float alpha) {
        for (int i = 0; i < prevArgs.length; ++i) {
            currArgs[i] = prevArgs[i] - alpha * target.partialDiffWith(i, prevArgs, dataset, label);
        }
        swapArgs();
    }
}

/**
 * 随机梯度下降求解器
 * \begin{aligned}
 * &\text{do}\\
 * &\quad\text{for each } (x,y) \text{ in } X  \\
 * &\quad\quad\theta_{i+1}=\theta_i-\alpha \nabla J(\theta_{i};\{(x,y)\}) \\
 * &\text{until } ||\theta_{i+1} - \theta_{i}||_{\infty}\le\beta \text{ OR iterate N times}\\
 * \end{aligned}
 */

final class SGDSolver extends Solver {

    public SGDSolver(TargetFunction target, float[][] dataset, float[] label) {
        super(target, dataset, label);
    }

    @Override
    protected void update(float alpha) {
        for (int i = 0; i < dataset.length; ++i) {
            float[][] sampleX = new float[][] { dataset[i] };
            float[] sampleY = new float[] { label[i] };
            for (int j = 0; j < prevArgs.length; ++j) {
                currArgs[j] = prevArgs[j] - alpha * target.partialDiffWith(j, prevArgs, sampleX, sampleY);
            }
            swapArgs();
        }
    }
}

/**
 * 小批量梯度下降求解器
 * \begin{aligned}
 * &\text{do}\\
 * &\quad \text{for each } b \text{ samples } \text{ in } X \text{ as } X'\\
 * &\quad\quad\theta_{i+1}=\theta_i-\alpha \nabla J(\theta_{i};X') \\
 * &\text{until } ||\theta_{i+1} - \theta_{i}||_{\infty}\le\beta \text{ OR iterate N times}
 * \end{aligned}
 */

final class MBGDSolver extends Solver {

    private int batch = 1;

    public MBGDSolver(TargetFunction target, float[][] dataset, float[] label) {
        super(target, dataset, label);
    }

    @Override
    protected void update(float alpha) {
        for (int i = 0; i < dataset.length; i += batch) {
            float[][] samplesX = Arrays.copyOfRange(dataset, i, Math.min(i + batch, dataset.length));
            float[] samplesY = Arrays.copyOfRange(label, i, Math.min(i + batch, label.length));
            for (int j = 0; j < prevArgs.length; ++j) {
                currArgs[j] = prevArgs[j] - alpha * target.partialDiffWith(j, prevArgs, samplesX, samplesY);
            }
            swapArgs();
        }
    }

    public void solve(float alpha, float beta, int numIterations, int batch) {
        this.batch = batch;
        super.solve(alpha, beta, numIterations);
    }
}

/**
 * Momentum梯度下降求解器
 * \begin{aligned}
 * &\text{do}\\
 * &\quad\text{for each } (x,y) \text{ in } X  \\
 * &\quad\quad\Delta\theta=\gamma\Delta\theta -\alpha \nabla J(\theta_{i};\{(x,y)\}) \\
 * &\quad\quad\theta_{i+1}=\theta_i+\Delta\theta\\
 * &\text{until } ||\theta_{i+1} - \theta_{i}||_{\infty}\le\beta \text{ OR iterate N times}\\
 * \end{aligned}
 */

final class MGDSolver extends Solver {

    private float gamma = 0.0f;

    private float[] deltaArgs;

    public MGDSolver(TargetFunction target, float[][] dataset, float[] label) {
        super(target, dataset, label);
        deltaArgs = new float[5];
    }

    @Override
    protected void update(float alpha) {
        for (int i = 0; i < dataset.length; ++i) {
            float[][] sampleX = new float[][] { dataset[i] };
            float[] sampleY = new float[] { label[i] };
            for (int j = 0; j < prevArgs.length; ++j) {
                deltaArgs[j] = gamma * deltaArgs[j] - alpha * target.partialDiffWith(j, prevArgs, sampleX, sampleY);
                currArgs[j] = prevArgs[j] + deltaArgs[j];
            }
            swapArgs();
        }
    }

    public void solve(float alpha, float beta, int numIterations, float gamma) {
        this.gamma = gamma;
        super.solve(alpha, beta, numIterations);
    }
}

public class App {

    private static final int SIZE = 100;

    public static void main(String[] args) throws Exception {

        float[][] dataset = new float[SIZE][5];
        float[]   label   = new float[SIZE];

        BufferedReader reader = new BufferedReader(new FileReader("resources/data.txt"));
        for (int i = 0; i < SIZE; ++i) {
            String[] raw = reader.readLine().split(" ");
            for (int j = 0; j < 5; ++j) {
                dataset[i][j] = Float.valueOf(raw[j]);
            }
            label[i] = Math.signum(Float.valueOf(raw[raw.length - 1]));
        }
        reader.close();

        ModelFunction model = new ModelFunction();
        TargetFunction target = new TargetFunction(model);
        
        String format = "%10s\t:\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\n";
        System.out.printf("%10s\t:\t%8s\t%8s\t%8s\t%8s\t%8s\n", "Method", "x0", "x1", "x2", "x3", "x4");

        {
            BGDSlover solver = new BGDSlover(target, dataset, label);
            solver.solve(0.001f, 1e-3f, 100000);
            float[] result = solver.getResult();
            System.out.printf(format, "BGDSlover", result[0], result[1], result[2], result[3], result[4]);
        }

        {
            SGDSolver solver = new SGDSolver(target, dataset, label);
            solver.solve(0.001f, 1e-3f, 100000);
            float[] result = solver.getResult();
            System.out.printf(format, "SGDSolver", result[0], result[1], result[2], result[3], result[4]);
        }

        {
            MBGDSolver solver = new MBGDSolver(target, dataset, label);
            solver.solve(0.001f, 1e-3f, 100000, 5);
            float[] result = solver.getResult();
            System.out.printf(format, "MBGDSolver", result[0], result[1], result[2], result[3], result[4]);
        }

        {
            MGDSolver solver = new MGDSolver(target, dataset, label);
            solver.solve(0.001f, 1e-3f, 100000, 0.1f);
            float[] result = solver.getResult();
            System.out.printf(format, "MGDSolver", result[0], result[1], result[2], result[3], result[4]);
        }

    }
}
