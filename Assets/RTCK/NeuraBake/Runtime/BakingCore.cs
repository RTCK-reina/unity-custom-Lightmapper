using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;

namespace RTCK.NeuraBake.Runtime
{
    public class BakingCore : ILightmapRenderer
    {
        // RenderTexture ���� Texture2D �֕ϊ����郁�\�b�h
        public static Texture2D ConvertRenderTextureToTexture2D(RenderTexture rt)
        {
            if (rt == null)
            {
                Debug.LogError("RenderTexture is null");
                return null;
            }

            var currentRT = RenderTexture.active;
            try
            {
                RenderTexture.active = rt;
                Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
                tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                tex.Apply();
                return tex;
            }
            catch (Exception e)
            {
                Debug.LogError($"Error converting RenderTexture to Texture2D: {e.Message}");
                return null;
            }
            finally
            {
                RenderTexture.active = currentRT;
            }
        }

        // Texture2D �� PNG �t�@�C���Ƃ��ĕۑ����郁�\�b�h
        public static bool SaveTexture2DToPNG(Texture2D tex, string path)
        {
            if (tex == null)
            {
                Debug.LogError("Texture2D is null");
                return false;
            }

            try
            {
                byte[] pngData = tex.EncodeToPNG();
                if (pngData == null)
                {
                    Debug.LogError("Failed to encode texture to PNG");
                    return false;
                }

                File.WriteAllBytes(path, pngData);
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"Error saving PNG file: {e.Message}");
                return false;
            }
        }        private readonly NeuraBakeSettings settings;
        private readonly Light[] sceneLights;
        private readonly NeuraBakeEmissiveSurface[] emissiveSurfaces;
        private readonly ConcurrentDictionary<Material, MaterialProperties> materialCache = new ConcurrentDictionary<Material, MaterialProperties>();
        private readonly ConcurrentDictionary<Mesh, MeshDataCache> meshDataCache = new ConcurrentDictionary<Mesh, MeshDataCache>();

        // Cached property IDs for performance
        private static readonly int BaseColorPropertyId = Shader.PropertyToID("_BaseColor");
        private static readonly int ColorPropertyId = Shader.PropertyToID("_Color");
        private static readonly int MetallicPropertyId = Shader.PropertyToID("_Metallic");
        private static readonly int GlossinessPropertyId = Shader.PropertyToID("_Glossiness");
        private static readonly int SmoothnessPropertyId = Shader.PropertyToID("_Smoothness");
        private static readonly int MainTexPropertyId = Shader.PropertyToID("_MainTex");
        private static readonly int MetallicGlossMapPropertyId = Shader.PropertyToID("_MetallicGlossMap");
        private static readonly int BumpMapPropertyId = Shader.PropertyToID("_BumpMap");
        private static readonly int MainTexSTPropertyId = Shader.PropertyToID("_MainTex_ST");
        private static readonly int MaskMapPropertyId = Shader.PropertyToID("_MaskMap");

        private class SpatialCache
        {
            private readonly struct RaycastKey : IEquatable<RaycastKey>
            {
                public readonly Vector3 Origin;
                public readonly Vector3 Direction;
                public readonly float MaxDistance;
                private const float OriginPrecisionFactor = 100f;
                private const float DistancePrecisionFactor = 100f;

                public RaycastKey(Vector3 origin, Vector3 direction, float maxDistance)
                {
                    Origin = new Vector3(
                        Mathf.Round(origin.x * OriginPrecisionFactor) / OriginPrecisionFactor,
                        Mathf.Round(origin.y * OriginPrecisionFactor) / OriginPrecisionFactor,
                        Mathf.Round(origin.z * OriginPrecisionFactor) / OriginPrecisionFactor
                    );
                    Direction = direction.normalized;
                    MaxDistance = Mathf.Round(maxDistance * DistancePrecisionFactor) / DistancePrecisionFactor;
                }

                public bool Equals(RaycastKey other)
                {
                    return Origin.Equals(other.Origin) &&
                           Direction.Equals(other.Direction) &&
                           MaxDistance == other.MaxDistance;
                }
                public override bool Equals(object obj) => obj is RaycastKey other && Equals(other);
                public override int GetHashCode()
                {
                    unchecked
                    {
                        int hash = 17;
                        hash = hash * 23 + Origin.GetHashCode();
                        hash = hash * 23 + Direction.GetHashCode();
                        hash = hash * 23 + MaxDistance.GetHashCode();
                        return hash;
                    }
                }
            }
            
            // ���C���X���b�h�ł̂ݎg�p����L���b�V��
            private readonly ConcurrentDictionary<RaycastKey, bool> cache = new ConcurrentDictionary<RaycastKey, bool>();
            
            // ���O�v�Z���ꂽ���ʁi�X���b�h�Z�[�t�j
            private readonly ConcurrentDictionary<RaycastKey, bool> precalculatedResults = new ConcurrentDictionary<RaycastKey, bool>();
            
            // ���C���X���b�h�ł̂݌Ăяo���\
            public void PrecalculateRaycast(Vector3 origin, Vector3 direction, float maxDistance)
            {
                var key = new RaycastKey(origin, direction, maxDistance);
                bool result = Physics.Raycast(origin, direction, maxDistance);
                precalculatedResults.TryAdd(key, result);
            }

            // �X���b�h�Z�[�t��Raycast���\�b�h�i���C���X���b�h�łȂ��Ă��Ăяo���\�j
            public bool Raycast(Ray ray, float maxDistance)
            {
                var key = new RaycastKey(ray.origin, ray.direction, maxDistance);
                
                // ���O�v�Z���ʂ�����΂����Ԃ�
                if (precalculatedResults.TryGetValue(key, out bool result))
                {
                    return result;
                }
                
                // �ߎ��L�[��T���i�߂��ʒu������̃��C�L���X�g���ʂ��ė��p�j
                foreach (var kvp in precalculatedResults)
                {
                    if (Vector3.Distance(kvp.Key.Origin, key.Origin) < 0.1f &&
                        Vector3.Dot(kvp.Key.Direction, key.Direction) > 0.98f &&
                        Mathf.Abs(kvp.Key.MaxDistance - key.MaxDistance) < 0.1f)
                    {
                        return kvp.Value;
                    }
                }
                
                // �L���b�V���~�X�̏ꍇ�̓f�t�H���g�l��Ԃ��i��Q���Ȃ��j
                // ���񏈗����ł�Physics.Raycast���Ăяo���Ȃ����߁A���S�Ȓl��Ԃ�
                return false;
            }

            public void Clear()
            {
                cache.Clear();
                // ����: precalculatedResults�̓N���A���Ȃ�
            }
        }

        private readonly SpatialCache mainRaycastCache = new SpatialCache();

        private class MaterialProperties
        {
            public Material material;
            public Color BaseColor { get; }
            public float Metallic { get; }
            public float Roughness { get; }
            public Texture2D BaseColorMap { get; }
            public Texture2D MetallicGlossMap { get; }
            public Texture2D NormalMap { get; }
            public Vector4 MainTexST { get; }
            public float Smoothness { get; }
            public Vector2 uvScale = Vector2.one;
            public Vector2 uvOffset = Vector2.zero;

            public MaterialProperties(Material mat)
            {
                material = mat;
                BaseColor = mat.HasProperty("_BaseColor") ? mat.GetColor("_BaseColor") :
                            (mat.HasProperty("_Color") ? mat.GetColor("_Color") : Color.white);
                Metallic = mat.HasProperty("_Metallic") ? mat.GetFloat("_Metallic") : 0f;
                Smoothness = mat.HasProperty("_Glossiness") ? mat.GetFloat("_Glossiness") :
                             (mat.HasProperty("_Smoothness") ? mat.GetFloat("_Smoothness") : 0.5f);
                Roughness = 1f - Smoothness;
                BaseColorMap = mat.HasProperty("_MainTex") ? mat.GetTexture("_MainTex") as Texture2D : null;
                MetallicGlossMap = mat.HasProperty("_MetallicGlossMap") ? mat.GetTexture("_MetallicGlossMap") as Texture2D : null;
                NormalMap = mat.HasProperty("_BumpMap") ? mat.GetTexture("_BumpMap") as Texture2D : null;
                MainTexST = mat.HasProperty("_MainTex_ST") ? mat.GetVector("_MainTex_ST") : new Vector4(1, 1, 0, 0);
                uvScale = new Vector2(MainTexST.x, MainTexST.y);
                uvOffset = new Vector2(MainTexST.z, MainTexST.w);
            }

            public Color GetSampledBaseColor(Vector2 uv)
            {
                Vector2 tiledUv = new Vector2(uv.x * uvScale.x + uvOffset.x, uv.y * uvScale.y + uvOffset.y);
                Color texColor = Color.white;
                if (BaseColorMap != null)
                {
                    if (BaseColorMap.isReadable) texColor = BaseColorMap.GetPixelBilinear(Mathf.Repeat(tiledUv.x, 1f), Mathf.Repeat(tiledUv.y, 1f));
                    else Debug.LogWarning($"NeuraBake: Texture '{BaseColorMap.name}' on material '{material.name}' is not readable.");
                }
                return BaseColor * texColor;
            }

            public (float sampledMetallic, float sampledRoughness) GetSampledMetallicRoughness(Vector2 uv)
            {
                Vector2 tiledUv = new Vector2(uv.x * uvScale.x + uvOffset.x, uv.y * uvScale.y + uvOffset.y);
                float finalMetallic = Metallic;
                float currentSmoothness = Smoothness;

                if (MetallicGlossMap != null)
                {
                    if (MetallicGlossMap.isReadable)
                    {
                        Color metallicGlossSample = MetallicGlossMap.GetPixelBilinear(Mathf.Repeat(tiledUv.x, 1f), Mathf.Repeat(tiledUv.y, 1f));
                        if (IsMaskMap(material, MetallicGlossMap))
                        {
                            if (material.shader.name.Contains("HDRP"))
                            {
                                finalMetallic = metallicGlossSample.r;
                                currentSmoothness = metallicGlossSample.a;
                            }
                            else if (material.shader.name.Contains("URP"))
                            {
                                finalMetallic = metallicGlossSample.g;
                                currentSmoothness = metallicGlossSample.a;
                            }
                            else
                            {
                                finalMetallic *= metallicGlossSample.r;
                                currentSmoothness *= metallicGlossSample.a;
                            }
                        }
                        else
                        {
                            finalMetallic *= metallicGlossSample.r;
                            currentSmoothness *= metallicGlossSample.a;
                        }
                    }
                    else Debug.LogWarning($"NeuraBake: Texture '{MetallicGlossMap.name}' on material '{material.name}' is not readable.");
                }
                return (Mathf.Clamp01(finalMetallic), 1f - Mathf.Clamp01(currentSmoothness));
            }
        }

        private class MeshDataCache
        {
            public readonly Vector3[] Vertices;
            public readonly Vector3[] Normals;
            public readonly Vector2[] UVs;
            public readonly int[] Triangles;
            public MeshDataCache(Mesh mesh)
            {
                Vertices = mesh.vertices; 
                Normals = mesh.normals;
                UVs = (mesh.uv2 != null && mesh.uv2.Length == mesh.vertexCount) ? mesh.uv2 : mesh.uv;
                Triangles = mesh.triangles;
            }
        }

        private struct CachedAmbientInfo
        {
            public UnityEngine.Rendering.AmbientMode mode;
            public Material skybox;
            public Color ambientSkyColor;
            public Color ambientEquatorColor;
            public Color ambientGroundColor;
            public Color ambientLight;
            public float ambientIntensity;
        }

        private struct CachedLightInfo
        {
            public LightType type;
            public Vector3 position;
            public Vector3 forward;
            public Color color;
            public float intensity;
            public float range;
            public float spotAngle;
        }

        public BakingCore(NeuraBakeSettings bakeSettings)
        {
            settings = bakeSettings ?? throw new ArgumentNullException(nameof(bakeSettings));
            sceneLights = GameObject.FindObjectsOfType<Light>().Where(l => l.isActiveAndEnabled && l.intensity > 0 && l.gameObject.activeInHierarchy).ToArray();
            emissiveSurfaces = GameObject.FindObjectsOfType<NeuraBakeEmissiveSurface>()
                                         .Where(es => es.enabled && es.bakeEmissive && es.intensity > 0 && es.gameObject.activeInHierarchy)
                                         .ToArray();
        }

        private MaterialProperties GetMaterialProperties(Material material)
        {
            lock (cacheLock) 
            { 
                if (!materialCache.TryGetValue(material, out MaterialProperties props)) 
                { 
                    props = new MaterialProperties(material); 
                    materialCache[material] = props; 
                } 
                return props; 
            }
        }
        
        private MeshDataCache GetMeshData(Mesh mesh)
        {
            lock (cacheLock) 
            { 
                if (!meshDataCache.TryGetValue(mesh, out MeshDataCache data)) 
                { 
                    data = new MeshDataCache(mesh); 
                    meshDataCache[mesh] = data; 
                } 
                return data; 
            }
        }

        // ILightmapRenderer�C���^�[�t�F�[�X�̎���
        public async Task<Texture2D> RenderAsync(CancellationToken token, IProgress<(float percentage, string message)> progress)
        {
            return await BakeLightmapAsync(token, progress);
        }

        public async Task<Texture2D> BakeLightmapAsync(CancellationToken token, IProgress<(float percentage, string message)> progressReporter)
        {
            MeshRenderer[] renderers = GameObject.FindObjectsOfType<MeshRenderer>()
                .Where(r => r.enabled && r.gameObject.isStatic && r.gameObject.activeInHierarchy).ToArray();
            if (renderers.Length == 0)
            {
                progressReporter?.Report((1f, "�x�C�N�Ώۂ̐ÓIMeshRenderer�Ȃ�"));
                return null;
            }

            MeshRenderer targetRenderer = renderers[0];
            Material targetMaterial = targetRenderer.sharedMaterial;
            MeshFilter meshFilter = targetRenderer.GetComponent<MeshFilter>();
            if (meshFilter?.sharedMesh == null || targetMaterial == null)
            {
                progressReporter?.Report((1f, "�Ώۃ��b�V��/�}�e���A���Ȃ�"));
                return null;
            }

            Mesh mesh = meshFilter.sharedMesh;
            MeshDataCache meshData = GetMeshData(mesh);
            if (meshData.UVs == null || meshData.UVs.Length == 0)
            {
                progressReporter?.Report((1f, "�Ώۃ��b�V����UV�Ȃ�"));
                return null;
            }

            int textureWidth = settings.atlasSize;
            int textureHeight = settings.atlasSize;
            progressReporter?.Report((0.01f, "���C�g�}�b�v�����J�n..."));

            Texture2D lightmapTexture = null;
            Color[] finalPixels = new Color[textureWidth * textureHeight];

            try
            {
                // ���O��main�X���b�h��localToWorld, world-space���_/�@�����v�Z���Ă���
                Matrix4x4 localToWorld = targetRenderer.transform.localToWorldMatrix;
                Vector3[] worldVertices = new Vector3[meshData.Vertices.Length];
                for (int i = 0; i < meshData.Vertices.Length; i++)
                {
                    worldVertices[i] = localToWorld.MultiplyPoint3x4(meshData.Vertices[i]);
                }
                Vector3[] worldNormals = new Vector3[meshData.Normals.Length];
                for (int i = 0; i < meshData.Normals.Length; i++)
                {
                    worldNormals[i] = localToWorld.MultiplyVector(meshData.Normals[i]).normalized;
                }

                // ���O��main�X���b�h��Random�l�𐶐����Ă���
                Vector2[][] randomOffsetsPerPixel = null;
                if (settings.sampleCount > 1)
                {
                    randomOffsetsPerPixel = new Vector2[textureWidth * textureHeight][];
                    for (int i = 0; i < textureWidth * textureHeight; i++)
                    {
                        Vector2[] offsets = new Vector2[settings.sampleCount];
                        for (int s = 0; s < settings.sampleCount; s++)
                        {
                            offsets[s] = new Vector2(UnityEngine.Random.value - 0.5f, UnityEngine.Random.value - 0.5f);
                        }
                        randomOffsetsPerPixel[i] = offsets;
                    }
                }
                
                // emissive�T�[�t�F�X�̃T���v�����O�̂��߂̗����l�����O����
                Vector3[] precomputedBarycentricCoords = new Vector3[1000];
                for (int i = 0; i < 1000; i++) // �\���Ȑ��̎��O�v�Z��p��
                {
                    float r1 = Random.value; 
                    float r2 = Random.value; 
                    float sqrtR1 = Mathf.Sqrt(r1); 
                    precomputedBarycentricCoords[i] = new Vector3(1.0f - sqrtR1, sqrtR1 * (1.0f - r2), sqrtR1 * r2);
                }

                // MaterialProperties ���L���b�V��
                MaterialProperties cachedMaterialProps = GetMaterialProperties(targetMaterial);

                // �A���r�G���g�����L���b�V��
                var ambientInfo = new CachedAmbientInfo
                {
                    mode = RenderSettings.ambientMode,
                    skybox = RenderSettings.skybox,
                    ambientSkyColor = RenderSettings.ambientSkyColor,
                    ambientEquatorColor = RenderSettings.ambientEquatorColor,
                    ambientGroundColor = RenderSettings.ambientGroundColor,
                    ambientLight = RenderSettings.ambientLight,
                    ambientIntensity = RenderSettings.ambientIntensity
                };

                // Light�����L���b�V��
                var cachedLights = sceneLights.Select(l => new CachedLightInfo
                {
                    type = l.type,
                    position = l.transform.position,
                    forward = l.transform.forward,
                    color = l.color,
                    intensity = l.intensity,
                    range = l.range,
                    spotAngle = l.spotAngle
                }).ToArray();

                // �V���h�E�T���v�����������O�v�Z
                Dictionary<int, Vector3[]> shadowJitterDirections = new Dictionary<int, Vector3[]>();
                for (int samples = 1; samples <= settings.shadowSamples; samples++)
                {
                    Vector3[] directions = new Vector3[samples];
                    for (int i = 0; i < samples; i++)
                    {
                        Vector3 randomOffset = UnityEngine.Random.onUnitSphere * 0.05f;
                        directions[i] = randomOffset;
                    }
                    shadowJitterDirections[samples] = directions;
                }

                // ���C���X���b�h�Ń��C�L���X�g�����O�v�Z
                progressReporter?.Report((0.1f, "���C�L���X�g���O�v�Z��..."));

                // ��葽���̃T���v�����O�|�C���g�𐶐��i���b�V�����璼�ڃT���v�����O�j
                HashSet<Vector3> samplePositions = new HashSet<Vector3>();
                int positionSamplesPerTriangle = 2; // �e�O�p�`����N�̈ʒu���T���v��

                // ���b�V����̈ʒu����T���v�����O
                for (int i = 0; i < meshData.Triangles.Length / 3; i += 3)
                {
                    int v0 = meshData.Triangles[i];
                    int v1 = meshData.Triangles[i + 1];
                    int v2 = meshData.Triangles[i + 2];
                    
                    Vector3 worldPos0 = worldVertices[v0];
                    Vector3 worldPos1 = worldVertices[v1];
                    Vector3 worldPos2 = worldVertices[v2];
                    
                    for (int j = 0; j < positionSamplesPerTriangle; j++)
                    {
                        // �O�p�`��̃����_���ȓ_���擾
                        Vector3 bary = new Vector3(
                            UnityEngine.Random.value,
                            UnityEngine.Random.value,
                            UnityEngine.Random.value
                        );
                        
                        // ���K��
                        float sum = bary.x + bary.y + bary.z;
                        if (sum > 0)
                            bary /= sum;
                            
                        Vector3 worldPos = bary.x * worldPos0 + bary.y * worldPos1 + bary.z * worldPos2;
                        samplePositions.Add(worldPos);
                    }
                }

                // �V�[�����̃����_���Ȉʒu���ǉ�
                for (int i = 0; i < 100; i++)
                {
                    Vector3 randomPos = new Vector3(
                        UnityEngine.Random.Range(-10f, 10f),
                        UnityEngine.Random.Range(0f, 5f),
                        UnityEngine.Random.Range(-10f, 10f)
                    );
                    samplePositions.Add(randomPos);
                }

                // ���ׂĂ̌����ƃT���v�����O�ʒu�̑g�ݍ��킹�Ń��C�L���X�g���ʂ����O�v�Z
                int processedRays = 0;
                int totalRays = samplePositions.Count * shadowJitterDirections[settings.shadowSamples].Length * cachedLights.Length;
                    
                for (int lightIdx = 0; lightIdx < cachedLights.Length; lightIdx++)
                {
                    var light = cachedLights[lightIdx];
                    foreach (var samplePos in samplePositions)
                    {
                        if (light.type == LightType.Directional)
                        {
                            Vector3 lightDirection = -light.forward;
                            Vector3[] jitterArr = shadowJitterDirections[settings.shadowSamples];
                            for (int d = 0; d < jitterArr.Length; d++)
                            {
                                Vector3 dirOffset = jitterArr[d];
                                Vector3 jitteredDir = (lightDirection + dirOffset).normalized;
                                Vector3 normal = Vector3.up; // �P�����̂���
                                Vector3 rayOrigin = samplePos + normal * 0.001f;
                                mainRaycastCache.PrecalculateRaycast(rayOrigin, -jitteredDir, 2000f);
                                processedRays++;
                                if (processedRays % 100 == 0)
                                {
                                    float progress = 0.1f + 0.2f * (float)processedRays / totalRays;
                                    progressReporter?.Report((progress, $"���C�L���X�g���O�v�Z��: {processedRays}/{totalRays}"));
                                }
                            }
                        }
                        else if (light.type == LightType.Point || light.type == LightType.Spot)
                        {
                            Vector3 lightToPointVec = samplePos - light.position;
                            float distSqr = lightToPointVec.sqrMagnitude;
                            if (distSqr <= light.range * light.range)
                            {
                                Vector3 pointLightDir = -lightToPointVec.normalized;
                                Vector3 normal = Vector3.up; // �P�����̂���
                                Vector3 rayOrigin = samplePos + normal * 0.001f;
                                mainRaycastCache.PrecalculateRaycast(rayOrigin, -pointLightDir, lightToPointVec.magnitude);
                                processedRays++;
                                if (processedRays % 100 == 0)
                                {
                                    float progress = 0.1f + 0.2f * (float)processedRays / totalRays;
                                    progressReporter?.Report((progress, $"���C�L���X�g���O�v�Z��: {processedRays}/{totalRays}"));
                                }
                            }
                        }
                    }
                }

                progressReporter?.Report((0.3f, "���C�L���X�g���O�v�Z����"));

                // CPU���񏈗��p�X
                progressReporter?.Report((0.5f, "CPU���񏈗���..."));
                int processedPixelCount = 0;
                int totalValidPixelsToProcess = textureWidth * textureHeight;
                int reportInterval = Math.Max(1, totalValidPixelsToProcess / 100);
                var parallelOptions = new ParallelOptions { CancellationToken = token, MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount - 1) };
                
                // �X���b�h���ł��A�N�Z�X�\�ȃL���b�V�����ێ����邽��clear���Ȃ�
                // mainRaycastCache.Clear();

                await Task.Run(() =>
                {
                    int chunkSize = 32; 
                    int numChunksX = Mathf.CeilToInt((float)textureWidth / chunkSize); 
                    int numChunksY = Mathf.CeilToInt((float)textureHeight / chunkSize);
                    
                    Parallel.For(0, numChunksX * numChunksY, parallelOptions, (chunkIdx, loopState) =>
                    {
                        int cX = chunkIdx % numChunksX; 
                        int cY = chunkIdx / numChunksX;
                        int sX = cX * chunkSize; 
                        int sY = cY * chunkSize;
                        int eX = Math.Min(sX + chunkSize, textureWidth); 
                        int eY = Math.Min(sY + chunkSize, textureHeight);
                        
                        for (int curY = sY; curY < eY; curY++)
                        {
                            for (int curX = sX; curX < eX; curX++)
                            {
                                if (token.IsCancellationRequested)
                                {
                                    loopState.Stop();
                                    return;
                                }
                                int pxIdx = curY * textureWidth + curX;
                                Vector2 tcUv = new Vector2((curX + 0.5f) / textureWidth, (curY + 0.5f) / textureHeight);
                                
                                if (FindTriangleAndBarycentricCoords(meshData, tcUv, out int triIdx, out Vector3 bary))
                                {
                                    int v0 = meshData.Triangles[triIdx * 3]; 
                                    int v1 = meshData.Triangles[triIdx * 3 + 1]; 
                                    int v2 = meshData.Triangles[triIdx * 3 + 2];
                                    
                                    // ������main�X���b�h�Ōv�Z����world-space���W/�@�����g��
                                    Vector3 wp = InterpolateVector3(worldVertices[v0], worldVertices[v1], worldVertices[v2], bary);
                                    Vector3 wn = InterpolateVector3(worldNormals[v0], worldNormals[v1], worldNormals[v2], bary).normalized;
                                    Vector2 lmUV = InterpolateVector2(meshData.UVs[v0], meshData.UVs[v1], meshData.UVs[v2], bary);
                                    
                                    Color accCol = Color.black; 
                                    float accBNY = 0f; 
                                    int validBNSamples = 0;
                                    
                                    for (int s = 0; s < settings.sampleCount; ++s)
                                    {
                                        Vector2 matUV = tcUv;
                                        if (settings.sampleCount > 1)
                                        {
                                            Vector2 offset = randomOffsetsPerPixel[pxIdx][s];
                                            matUV = new Vector2(tcUv.x + offset.x / textureWidth, tcUv.y + offset.y / textureHeight);
                                        }
                                        var (pxCol, bnY, unocRay) = CalculatePixelColor(wp, wn, cachedMaterialProps, lmUV, matUV, ambientInfo, cachedLights, shadowJitterDirections, precomputedBarycentricCoords);
                                        accCol += pxCol;
                                        if (settings.directional && unocRay > 0)
                                        {
                                            accBNY += bnY * unocRay;
                                            validBNSamples += unocRay;
                                        }
                                    }
                                    
                                    Color fpxCol = accCol / settings.sampleCount;
                                    
                                    if (settings.directional) 
                                    { 
                                        if (validBNSamples > 0) 
                                            fpxCol.a = Mathf.Clamp01(0.5f + (accBNY / validBNSamples) * 0.5f); 
                                        else if (settings.useAmbientOcclusion && settings.aoSampleCount > 0) 
                                            fpxCol.a = 0f; 
                                        else 
                                            fpxCol.a = 0.5f; 
                                    }
                                    else 
                                    { 
                                        fpxCol.a = 1f; 
                                    }
                                    
                                    finalPixels[pxIdx] = fpxCol;
                                }
                                else 
                                { 
                                    finalPixels[pxIdx] = Color.clear; 
                                }
                                
                                int curProc = Interlocked.Increment(ref processedPixelCount);
                                if (curProc % reportInterval == 0 || curProc == totalValidPixelsToProcess)
                                {
                                    float prog = 0.5f + (0.5f * curProc / totalValidPixelsToProcess);
                                    progressReporter?.Report((prog, $"CPU �s�N�Z��������: {curProc}/{totalValidPixelsToProcess}"));
                                }
                            }
                        }
                    });
                }, token);
                
                if (token.IsCancellationRequested) 
                    throw new OperationCanceledException();

                DilationEdgePadding(finalPixels, textureWidth, textureHeight, 8); // �G�b�W�p�f�B���O

                lightmapTexture = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBAHalf, false, true);
                lightmapTexture.name = $"{targetRenderer.gameObject.name}_Lightmap_Baked_CPU";
                lightmapTexture.wrapMode = TextureWrapMode.Clamp;
                lightmapTexture.filterMode = FilterMode.Bilinear;
                lightmapTexture.SetPixels(finalPixels);
                lightmapTexture.Apply(true, false); //�~�b�v�}�b�v�͕s�v�Ȃ̂�false, �ǂݎ��s�ɂ͂��Ȃ�
                progressReporter?.Report((1f, "CPU���񏈗�����"));

                if (lightmapTexture == null && !token.IsCancellationRequested)
                {
                    progressReporter?.Report((1f, "���C�g�}�b�v�������s")); 
                    return null;
                }

                // �f�m�C�U�[�K�p
                if (settings.useDenoiser)
                {
                    progressReporter?.Report((0.9f, "�f�m�C�U�[�K�p��..."));
                    progressReporter?.Report((1f, "�f�m�C�U�[�K�p����"));
                }

                return lightmapTexture;
            }
            catch (OperationCanceledException) 
            { 
                progressReporter?.Report((0f, "�����L�����Z��")); 
                Debug.Log("BakingCore: �����L�����Z��"); 
                return null; 
            }
            catch (Exception ex) 
            { 
                progressReporter?.Report((0f, $"�G���[: {ex.GetType().Name}")); 
                Debug.LogException(ex); 
                return null; 
            }
        }

        private bool FindTriangleAndBarycentricCoords(MeshDataCache meshData, Vector2 uv, out int triangleStartVertexIndex, out Vector3 barycentricCoords)
        {
            triangleStartVertexIndex = 0; 
            barycentricCoords = Vector3.zero;
            for (int i = 0; i < meshData.Triangles.Length / 3; i++)
            {
                Vector2 uv0 = meshData.UVs[meshData.Triangles[i * 3 + 0]]; 
                Vector2 uv1 = meshData.UVs[meshData.Triangles[i * 3 + 1]]; 
                Vector2 uv2 = meshData.UVs[meshData.Triangles[i * 3 + 2]];
                barycentricCoords = CalculateBarycentricCoords(uv, uv0, uv1, uv2);
                if (barycentricCoords.x >= -1e-4f && barycentricCoords.y >= -1e-4f && barycentricCoords.z >= -1e-4f && 
                    barycentricCoords.x <= 1.0001f && barycentricCoords.y <= 1.0001f && barycentricCoords.z <= 1.0001f)
                {
                    triangleStartVertexIndex = i; 
                    return true;
                }
            }
            return false;
        }
        
        private Vector3 CalculateBarycentricCoords(Vector2 p, Vector2 a, Vector2 b, Vector2 c)
        {
            Vector2 v0 = b - a, v1 = c - a, v2 = p - a; 
            float d00 = Vector2.Dot(v0, v0); 
            float d01 = Vector2.Dot(v0, v1); 
            float d11 = Vector2.Dot(v1, v1); 
            float d20 = Vector2.Dot(v2, v0); 
            float d21 = Vector2.Dot(v2, v1);
            float denom = d00 * d11 - d01 * d01; 
            if (Mathf.Abs(denom) < 1e-6f) return new Vector3(-1, -1, -1);
            float v_ = (d11 * d20 - d01 * d21) / denom; 
            float w_ = (d00 * d21 - d01 * d20) / denom; 
            return new Vector3(1.0f - v_ - w_, v_, w_);
        }
        
        private Vector3 InterpolateVector3(Vector3 v0, Vector3 v1, Vector3 v2, Vector3 barycentric) 
        { 
            return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z; 
        }
        
        private Vector2 InterpolateVector2(Vector2 v0, Vector2 v1, Vector2 v2, Vector3 barycentric) 
        { 
            return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z; 
        }

        private (float occlusion, float bentNormalY, int unoccludedRays) CalculateAmbientOcclusionAndBentNormal(Vector3 position, Vector3 normal, int sampleCount)
        {
            if (sampleCount <= 0) return (0f, normal.y, 0);

            float sumOcclusionFactor = 0f;
            Vector3 accumulatedUnoccludedNormal = Vector3.zero;
            int unoccludedRayCountAtomic = 0;
            float maxAoDistance = 2.0f;

            // AO�T���v�����������O�����i�P�ʔ�����ɕ��z�j
            Vector3[] sampleDirections = GenerateHemisphereDirections(normal, sampleCount);

            Parallel.For(0, sampleCount, i =>
            {
                Ray aoRay = new Ray(position + normal * 0.001f, sampleDirections[i]);
                if (!mainRaycastCache.Raycast(aoRay, maxAoDistance))
                {
                    lock (cacheLock) { accumulatedUnoccludedNormal += sampleDirections[i]; }
                    Interlocked.Increment(ref unoccludedRayCountAtomic);
                }
                else
                {
                    lock (cacheLock) { sumOcclusionFactor += 1f; }
                }
            });

            float finalOcclusion = (sampleCount > 0) ? sumOcclusionFactor / sampleCount : 0f;
            float finalBentNormalY = normal.y;
            if (unoccludedRayCountAtomic > 0)
            {
                finalBentNormalY = (accumulatedUnoccludedNormal / unoccludedRayCountAtomic).normalized.y;
            }
            return (finalOcclusion, finalBentNormalY, unoccludedRayCountAtomic);
        }

        // ���C���X���b�h�Ŏg�p����A������ɂقڋϓ��ɕ��z����x�N�g���𐶐����郁�\�b�h
        private Vector3[] GenerateHemisphereDirections(Vector3 normal, int count)
        {
            Vector3[] directions = new Vector3[count];
            
            // �ȒP�ȃt�B�{�i�b�`���ʕ��z���g�p
            float goldenRatio = (1f + Mathf.Sqrt(5f)) / 2f;
            float angleIncrement = Mathf.PI * 2f * goldenRatio;
            
            for (int i = 0; i < count; i++)
            {
                float t = (float)i / count;
                float inclination = Mathf.Acos(1f - t); // �������z�̂���1����X�^�[�g
                float azimuth = angleIncrement * i;
                
                // ���ʍ��W���璼�����W�֕ϊ�
                float x = Mathf.Sin(inclination) * Mathf.Cos(azimuth);
                float y = Mathf.Sin(inclination) * Mathf.Sin(azimuth);
                float z = Mathf.Cos(inclination);
                Vector3 dir = new Vector3(x, z, y); // Unity�̍��W�n�ɍ��킹��
                
                // normal��������Ƃ��Ē���
                Vector3 up = Vector3.up;
                if (Mathf.Abs(Vector3.Dot(normal, up)) > 0.99f)
                    up = Vector3.forward;
                    
                Quaternion rotation = Quaternion.FromToRotation(Vector3.up, normal);
                directions[i] = rotation * dir;
            }
            
            return directions;
        }

        private (Color color, float bentNormalY, int unoccludedRays) CalculatePixelColor(
            Vector3 worldPos, Vector3 worldNormal, MaterialProperties matProps,
            Vector2 lightmapUV, Vector2 materialSampleUV, CachedAmbientInfo ambientInfo, CachedLightInfo[] cachedLights,
            Dictionary<int, Vector3[]> shadowJitterDirections, Vector3[] precomputedBarycentricCoords)
        {
            Color albedo = matProps.GetSampledBaseColor(materialSampleUV);
            var (metallic, roughness) = matProps.GetSampledMetallicRoughness(materialSampleUV);

            Color finalColor = Color.black;

            Color skyContribution = CalculateSkyLight(worldPos, worldNormal, albedo, ambientInfo);
            finalColor += skyContribution;

            foreach (var light in cachedLights)
            {
                Vector3 lightDir;
                Color lightColorAtPoint = light.color * light.intensity;
                float attenuation = 1.0f;

                if (light.type == LightType.Directional)
                {
                    lightDir = -light.forward;
                }
                else
                {
                    Vector3 lightToPointVec = worldPos - light.position;
                    float distSqr = lightToPointVec.sqrMagnitude;
                    if (distSqr > light.range * light.range) continue;
                    float dist = Mathf.Sqrt(distSqr);
                    lightDir = -lightToPointVec.normalized;
                    attenuation = 1.0f / (1.0f + 25.0f * distSqr / (light.range * light.range));
                    if (light.type == LightType.Spot)
                    {
                        float spotAngle = Vector3.Angle(light.forward, -lightDir);
                        if (spotAngle > light.spotAngle / 2f) continue;
                        float outerConeRad = (light.spotAngle / 2f) * Mathf.Deg2Rad;
                        float cosOuter = Mathf.Cos(outerConeRad);
                        float cosAngle = Vector3.Dot(light.forward, -lightDir);
                        float spotFalloff = Mathf.Pow(Mathf.Clamp01((cosAngle - cosOuter) / (1f - cosOuter + 1e-4f)), 2f);
                        attenuation *= spotFalloff;
                    }
                }

                float maxShadowDist = (light.type == LightType.Directional) ? 2000f : Vector3.Distance(worldPos, light.position);
                float shadowFactor = CalculateShadowFactor(worldPos, worldNormal, lightDir, maxShadowDist, settings.shadowSamples, shadowJitterDirections);

                if (shadowFactor > 0)
                {
                    Vector3 viewDir = worldNormal;
                    Color directLight = CalculatePBRDirectLight(lightDir, viewDir, worldNormal, lightColorAtPoint * attenuation * shadowFactor, albedo, metallic, roughness);
                    finalColor += directLight;
                }
            }

            if (emissiveSurfaces.Length > 0)
            {
                finalColor += CalculateEmissiveContribution(worldPos, worldNormal, albedo, precomputedBarycentricCoords);
            }

            float occlusionFactor = 0f;
            float bentNormalYComponent = worldNormal.y;
            int unoccludedRayCount = 0;

            if (settings.useAmbientOcclusion && settings.aoSampleCount > 0)
            {
                (occlusionFactor, bentNormalYComponent, unoccludedRayCount) = CalculateAmbientOcclusionAndBentNormal(worldPos, worldNormal, settings.aoSampleCount);
                finalColor *= (1f - occlusionFactor);
            }
            else if (settings.directional)
            {
                unoccludedRayCount = 1; // For bent normal calculation, even if AO is off
            }

            return (finalColor, bentNormalYComponent, unoccludedRayCount);
        }
        
        private Color CalculateEmissiveContribution(Vector3 worldPos, Vector3 worldNormal, Color albedo, Vector3[] precomputedBarycentricCoords)
        {
            Color totalEmissiveContribution = Color.black;
            for (int e = 0; e < emissiveSurfaces.Length; e++)
            {
                NeuraBakeEmissiveSurface emissiveSurface = emissiveSurfaces[e];
                MeshFilter emissiveMeshFilter = emissiveSurface.GetComponent<MeshFilter>();
                if (emissiveMeshFilter?.sharedMesh == null) continue;
                Mesh emissiveMesh = emissiveMeshFilter.sharedMesh;
                int emissiveSamplePoints = 16;
                Color accumulatedLightFromThisSurface = Color.black;
                for (int i = 0; i < emissiveSamplePoints; i++)
                {
                    Vector3 b = precomputedBarycentricCoords[i % precomputedBarycentricCoords.Length];
                    int randomTriangleStartIdx = (i * 17 + (int)(worldPos.x * 13 + worldPos.y * 7 + worldPos.z * 31)) % (emissiveMesh.triangles.Length / 3);
                    randomTriangleStartIdx *= 3;
                    if (randomTriangleStartIdx + 2 >= emissiveMesh.triangles.Length) continue;
                    int vIdx0 = emissiveMesh.triangles[randomTriangleStartIdx + 0];
                    int vIdx1 = emissiveMesh.triangles[randomTriangleStartIdx + 1];
                    int vIdx2 = emissiveMesh.triangles[randomTriangleStartIdx + 2];
                    if (vIdx0 >= emissiveMesh.vertexCount || vIdx1 >= emissiveMesh.vertexCount || vIdx2 >= emissiveMesh.vertexCount) continue;
                    Vector3 emissivePointLocal = InterpolateVector3(emissiveMesh.vertices[vIdx0], emissiveMesh.vertices[vIdx1], emissiveMesh.vertices[vIdx2], b);
                    Vector3 emissiveNormalLocal = InterpolateVector3(emissiveMesh.normals[vIdx0], emissiveMesh.normals[vIdx1], emissiveMesh.normals[vIdx2], b).normalized;
                    Vector3 emissivePointWorld = emissiveSurface.transform.TransformPoint(emissivePointLocal);
                    Vector3 emissiveNormalWorld = emissiveSurface.transform.TransformDirection(emissiveNormalLocal).normalized;
                    Vector3 dirToEmissive = emissivePointWorld - worldPos;
                    float distToEmissiveSqr = dirToEmissive.sqrMagnitude;
                    if (distToEmissiveSqr < 1e-5f) continue;
                    float distToEmissive = Mathf.Sqrt(distToEmissiveSqr);
                    Vector3 normalizedDirToEmissive = dirToEmissive / distToEmissive;
                    float NdotL_emissive = Mathf.Max(0, Vector3.Dot(worldNormal, normalizedDirToEmissive));
                    float NdotL_source = Mathf.Max(0, Vector3.Dot(emissiveNormalWorld, -normalizedDirToEmissive));
                    if (NdotL_emissive > 0 && NdotL_source > 0)
                    {
                        Ray visibilityRay = new Ray(worldPos + worldNormal * 0.001f, normalizedDirToEmissive);
                        if (!mainRaycastCache.Raycast(visibilityRay, distToEmissive - 0.002f))
                        {
                            Color emissiveLightColor = emissiveSurface.emissiveColor * emissiveSurface.intensity * settings.emissiveBoost;
                            float formFactorApprox = (NdotL_emissive * NdotL_source) / (Mathf.PI * distToEmissiveSqr + 1e-4f);
                            Color contribution = (albedo / Mathf.PI) * emissiveLightColor * formFactorApprox;
                            accumulatedLightFromThisSurface += contribution;
                        }
                    }
                }
                totalEmissiveContribution += accumulatedLightFromThisSurface / emissiveSamplePoints;
            }
            return totalEmissiveContribution;
        }
        
        private Color CalculateSkyLight(Vector3 worldPos, Vector3 worldNormal, Color albedo, CachedAmbientInfo ambientInfo)
        {
            Color skyColor;
            if (ambientInfo.mode == UnityEngine.Rendering.AmbientMode.Skybox && ambientInfo.skybox != null)
            {
                skyColor = ambientInfo.ambientSkyColor;
            }
            else if (ambientInfo.mode == UnityEngine.Rendering.AmbientMode.Trilight)
            {
                float dotUp = Vector3.Dot(worldNormal, Vector3.up);
                float upLerp = Mathf.Clamp01(dotUp);
                float downLerp = Mathf.Clamp01(-dotUp);
                skyColor = Color.Lerp(ambientInfo.ambientEquatorColor, ambientInfo.ambientSkyColor, upLerp);
                skyColor = Color.Lerp(skyColor, ambientInfo.ambientGroundColor, downLerp);
            }
            else if (ambientInfo.mode == UnityEngine.Rendering.AmbientMode.Flat)
            {
                skyColor = ambientInfo.ambientLight;
            }
            else
            {
                skyColor = ambientInfo.ambientSkyColor;
            }
            return skyColor * albedo * settings.skyIntensity * ambientInfo.ambientIntensity;
        }
        
        // PBR���C�e�B���O�v�Z
        private Color CalculatePBRDirectLight(Vector3 L, Vector3 V, Vector3 N, Color lightColor, Color albedo, float metallic, float roughness)
        {
            L = L.normalized; 
            V = V.normalized; 
            N = N.normalized;
            Vector3 H = (L + V).normalized;
            float NdotL = Mathf.Max(0f, Vector3.Dot(N, L));
            if (NdotL <= 0f) return Color.black;
            
            float NdotV = Mathf.Max(0f, Vector3.Dot(N, V));
            float alpha = roughness * roughness;
            
            // F0�̌v�Z�i�����x�Ɋ�Â��j
            Color F0 = Color.Lerp(new Color(0.04f, 0.04f, 0.04f), albedo, metallic);
            
            // �eBRDF�R���|�[�l���g�̌v�Z
            float D = GGX_Distribution(N, H, alpha);
            Color F = Fresnel_Schlick(Mathf.Max(0f, Vector3.Dot(H, V)), F0);
            float Vis = Smith_Visibility_JointGGX(N, V, L, alpha);
            
            // �G�l���M�[�ۑ����l�������g�U����
            Color kd = Color.Lerp(Color.white - F, Color.black, metallic);
            Color diffuse = kd * albedo / Mathf.PI;
            
            // ���ʔ���
            Color specular = D * F * Vis;
            
            return (diffuse + specular) * lightColor * NdotL;
        }

        private float GGX_Distribution(Vector3 N, Vector3 H, float alpha)
        {
            float NdotH = Mathf.Max(0f, Vector3.Dot(N, H));
            float alphaSq = alpha * alpha;
            float denom = NdotH * NdotH * (alphaSq - 1f) + 1f;
            return alphaSq / (Mathf.PI * denom * denom + 1e-7f);
        }

        private Color Fresnel_Schlick(float cosTheta, Color F0)
        {
            return F0 + (Color.white - F0) * Mathf.Pow(1f - Mathf.Max(0, cosTheta), 5f);
        }

        private float Smith_Visibility_JointGGX(Vector3 N, Vector3 V, Vector3 L, float alpha)
        {
            float NdotV = Mathf.Max(Vector3.Dot(N, V), 0.0f) + 1e-5f;
            float NdotL = Mathf.Max(Vector3.Dot(N, L), 0.0f) + 1e-5f;
            float roughness_sq = alpha * alpha;
            float G_SmithV = (2.0f * NdotV) / (NdotV + Mathf.Sqrt(roughness_sq + (1.0f - roughness_sq) * NdotV * NdotV));
            float G_SmithL = (2.0f * NdotL) / (NdotL + Mathf.Sqrt(roughness_sq + (1.0f - roughness_sq) * NdotL * NdotL));
            return G_SmithV * G_SmithL;
        }

        private static bool IsMaskMap(Material material, Texture2D texture) 
        { 
            if (texture == null || material?.shader == null) return false; 
            bool isHDRP = material.shader.name.Contains("HDRP") || material.shader.name.Contains("High Definition"); 
            bool isURP = material.shader.name.Contains("URP") || material.shader.name.Contains("Universal"); 
            bool hasMaskMapProperty = material.HasProperty("_MaskMap"); 
            return isHDRP || hasMaskMapProperty || (texture.name.Contains("_MaskMap") && isURP); 
        }
        
        private Color ApplyColorSpaceCorrection(Color color) 
        { 
            if (QualitySettings.activeColorSpace == ColorSpace.Gamma) 
                return new Color(Mathf.LinearToGammaSpace(color.r), Mathf.LinearToGammaSpace(color.g), Mathf.LinearToGammaSpace(color.b), color.a); 
            return color; 
        }
        
        private float CalculateShadowFactor(Vector3 worldPos, Vector3 worldNormal, Vector3 lightDir, 
                                           float maxShadowDist, int shadowSamples, 
                                           Dictionary<int, Vector3[]> jitterDirections)
        {
            if (shadowSamples <= 0) return 1.0f;
            float shadowAccumulator = 0f;
            
            if (shadowSamples == 1)
            { 
                Ray sr = new Ray(worldPos + worldNormal * 0.001f, -lightDir); 
                return mainRaycastCache.Raycast(sr, maxShadowDist - 0.002f) ? 0.0f : 1.0f; 
            }
            
            float[] sampleResults = new float[shadowSamples];
            
            // ���O�v�Z�ς݂̕����f�[�^���g�p
            Vector3[] randomDirections = jitterDirections[shadowSamples];
            
            Parallel.For(0, shadowSamples, i =>
            {
                Vector3 jld = lightDir;
                if (shadowSamples > 1)
                {
                    // ���O�������ꂽ�����l���g�p
                    jld = (lightDir + randomDirections[i]).normalized;
                }
                Ray sr = new Ray(worldPos + worldNormal * 0.001f, -jld);
                sampleResults[i] = mainRaycastCache.Raycast(sr, maxShadowDist - 0.002f) ? 0.0f : 1.0f;
            });
            
            for (int i = 0; i < shadowSamples; i++)
                shadowAccumulator += sampleResults[i];
                
            return shadowAccumulator / shadowSamples;
        }

        // ���ǂ��ꂽ�G�b�W�p�f�B���O���� - UV�����l�����ċ��E���̔����݂�h�~
        private static void DilationEdgePadding(Color[] pixels, int width, int height, int iterations = 2)
        {
            if (pixels == null || pixels.Length != width * height) return;

            // ���̃s�N�Z����ۑ��i�ŏI�I�ȏ����Ō��̗L���ȃs�N�Z����ێ����邽�߁j
            Color[] originalPixels = new Color[pixels.Length];
            Array.Copy(pixels, originalPixels, pixels.Length);

            // �s�N�Z���̏�����Ԃ�ǐՂ���}�X�N (0=������, 1=�L���ȃs�N�Z��, 2=�p�f�B���O���ꂽ�s�N�Z��)
            byte[] pixelState = new byte[pixels.Length];
            for (int i = 0; i < pixels.Length; i++)
            {
                if (pixels[i].a > 0.001f) 
                {
                    pixelState[i] = 1; // ���ɗL���ȃs�N�Z��
                }
            }

            // UV���̌��̈�����ʁi�אڂ���L���ȃs�N�Z���͓������ɑ�����j
            int[] islandIds = new int[pixels.Length];
            int currentIslandId = 0;

            // �e�L���ȃs�N�Z������n�߂ĘA�����������ʁi�t���b�h�t�B���@�j
            for (int i = 0; i < pixels.Length; i++)
            {
                if (pixelState[i] == 1 && islandIds[i] == 0)
                {
                    currentIslandId++;
                    Queue<int> queue = new Queue<int>();
                    queue.Enqueue(i);
                    islandIds[i] = currentIslandId;

                    while (queue.Count > 0)
                    {
                        int pixelIdx = queue.Dequeue();
                        int x = pixelIdx % width;
                        int y = pixelIdx / width;

                        // 4�����܂���8�����̗אڃs�N�Z�����`�F�b�N
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                // 4�����݂̂��g���ꍇ�͂��̃`�F�b�N������
                                // if (dx != 0 && dy != 0) continue; // �΂ߕ������X�L�b�v

                                int nx = x + dx;
                                int ny = y + dy;
                                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                                {
                                    int neighborIdx = ny * width + nx;
                                    if (pixelState[neighborIdx] == 1 && islandIds[neighborIdx] == 0)
                                    {
                                        islandIds[neighborIdx] = currentIslandId;
                                        queue.Enqueue(neighborIdx);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ������̊g������
            for (int iter = 0; iter < iterations; iter++)
            {
                Color[] tempPixels = new Color[pixels.Length];
                Array.Copy(pixels, tempPixels, pixels.Length);
                byte[] newPixelState = new byte[pixelState.Length];
                Array.Copy(pixelState, newPixelState, pixelState.Length);

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int currentIdx = y * width + x;
                        
                        // �������̃s�N�Z���̂݃p�f�B���O�ΏۂƂ���
                        if (pixelState[currentIdx] != 0) continue;
                        
                        // �eUV�����ƂɐF�̒~�ς����Z�b�g
                        Dictionary<int, ColorAccumulator> islandContributions = new Dictionary<int, ColorAccumulator>();
                        
                        // ����8�s�N�Z������̊�^��~��
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                if (dx == 0 && dy == 0) continue; // �������g�̓X�L�b�v
                                
                                int nx = x + dx;
                                int ny = y + dy;
                                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                                {
                                    int neighborIdx = ny * width + nx;
                                    // �L���ȐF�����s�N�Z���̂ݍl��
                                    if (pixelState[neighborIdx] > 0)
                                    {
                                        int islandId = islandIds[neighborIdx];
                                        if (islandId > 0) // �L����UV���ɑ����Ă���
                                        {
                                            // ���̓�����̊�^���܂��ǐՂ��Ă��Ȃ���Ώ�����
                                            if (!islandContributions.TryGetValue(islandId, out ColorAccumulator accumulator))
                                            {
                                                accumulator = new ColorAccumulator();
                                                islandContributions[islandId] = accumulator;
                                            }
                                            
                                            // �����ɉ������d�ݕt��
                                            float weight = (dx == 0 || dy == 0) ? 1.0f : 0.7071f; // �΂߂́�2�̋����Ȃ̂ŏd�݂����炷
                                            
                                            // �F�̒~��
                                            accumulator.AddColor(pixels[neighborIdx], weight);
                                        }
                                    }
                                }
                            }
                        }
                        
                        // �ł���^�̑傫��UV����I������
                        int bestIslandId = 0;
                        float bestWeight = 0f;
                        Color bestColor = Color.clear;
                        
                        foreach (var pair in islandContributions)
                        {
                            if (pair.Value.TotalWeight > bestWeight)
                            {
                                bestIslandId = pair.Key;
                                bestWeight = pair.Value.TotalWeight;
                                bestColor = pair.Value.GetAverageColor();
                            }
                        }
                        
                        // �\���Ȋ�^������΁A���̃s�N�Z���ɐF�����蓖�Ă�
                        if (bestWeight > 0.01f)
                        {
                            tempPixels[currentIdx] = bestColor;
                            newPixelState[currentIdx] = 2; // �p�f�B���O���ꂽ�s�N�Z���Ƃ��ă}�[�N
                            islandIds[currentIdx] = bestIslandId; // ���̓��ɑ�����ƃ}�[�N
                        }
                    }
                }
                
                // �X�V�����C���o�b�t�@�ɔ��f
                Array.Copy(tempPixels, pixels, pixels.Length);
                Array.Copy(newPixelState, pixelState, pixelState.Length);
            }
            
            // ���X�L���������s�N�Z���̐F��ێ�
            for (int i = 0; i < pixels.Length; i++)
            {
                if (pixelState[i] == 1) // ���X�L���������s�N�Z��
                {
                    pixels[i] = originalPixels[i];
                }
                else if (pixelState[i] == 2) // �p�f�B���O���ꂽ�s�N�Z��
                {
                    // �A���t�@�l��1.0�ɐݒ�i�����łȂ����Ƃ�ۏ؁j
                    pixels[i].a = 1.0f;
                }
            }
        }

        // �F�̒~�ςƕ��ω����s���w���p�[�N���X
        private class ColorAccumulator
        {
            private float r = 0f, g = 0f, b = 0f, a = 0f;
            public float TotalWeight { get; private set; } = 0f;
            
            public void AddColor(Color color, float weight)
            {
                r += color.r * weight;
                g += color.g * weight;
                b += color.b * weight;
                a += color.a * weight;
                TotalWeight += weight;
            }
            
            public Color GetAverageColor()
            {
                if (TotalWeight > 0f)
                {
                    return new Color(
                        r / TotalWeight,
                        g / TotalWeight,
                        b / TotalWeight,
                        a / TotalWeight
                    );
                }
                return Color.clear;
            }
        }
    }
}